import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip import create_model_and_transforms, get_tokenizer
from autoswin.q_former import QFormer
from autoswin.swin_decoder import SwinTransformerV2Decoder, make_decoder_config
from timm.models.swin_transformer_v2 import SwinTransformerV2
from torchvision import transforms
from autoswin.ptuned_text_encoder import PEFTTextEncoder


# Foundation model configurations
FOUNDATION_MODELS = {
    "biomedclip": {
        "model_name": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "embed_dim": 512,
        "image_size": 224,
    },
    "medclip": {
        "model_name": "hf-hub:luhuitong/CLIP-ViT-L-14-448px-MedICaT-ROCO",
        "embed_dim": 768,  # ViT-L uses 768-dim embeddings
        "image_size": 448,
    },
}


def get_foundation_model_config(model_key: str):
    """Get foundation model configuration.
    
    Args:
        model_key: 'biomedclip' or 'medclip'
        
    Returns:
        dict with model_name, embed_dim, image_size
        
    Raises:
        ValueError: if model_key is not recognized
    """
    if model_key not in FOUNDATION_MODELS:
        valid = ", ".join(FOUNDATION_MODELS.keys())
        raise ValueError(f"Unknown foundation model '{model_key}'. Must be one of: {valid}")
    return FOUNDATION_MODELS[model_key]


def weights_changed(model, old_state_dict):
    for key in model.state_dict():
        if key in old_state_dict:
            if not torch.equal(model.state_dict()[key], old_state_dict[key]):
                return True
    return False


class LearnableClipNorm(nn.Module):
    """
    Learnable affine normalization initialized to CLIP mean/std.
    This bridges the distribution between the Swin adapter output and
    the (possibly frozen) CLIP visual encoder.

    It keeps the same behavior as fixed CLIP normalization at init,
    but can adapt during training.
    """

    def __init__(
        self,
        init_mean=(0.48145466, 0.4578275, 0.40821073),
        init_std=(0.26862954, 0.26130258, 0.27577711),
    ):
        super().__init__()
        self.mean = nn.Parameter(
            torch.tensor(init_mean, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.std = nn.Parameter(
            torch.tensor(init_std, dtype=torch.float32).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + 1e-6)


class SwinAutoencoderWithAdapter(nn.Module):
    """Swin encoder with per-stage adapters and a multi-entry decoder.

    Supports progressive training by exiting at intermediate encoder
    stages and computing fidelity loss at the actual output resolution.
    """

    def __init__(self,
                 img_size=896,
                 patch_size=2,
                 in_chans=1,
                 embed_dim=48,
                 depths=[2, 2, 2],
                 num_heads=[2, 4, 8],
                 window_size=4,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 drop_rate=0,
                 attn_drop_rate=0,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=True,
                 num_classes=0,
                 use_decoder=False,
                 out_chans=1,
                 decoder_depth=2,
                 decoder_window_size=4,
                 ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.use_decoder = use_decoder

        self.swin = SwinTransformerV2(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            num_classes=num_classes,
        )

        # Per-stage adapters: each maps the encoder output at that stage to 3 channels.
        # Stage i has embed_dim * 2**i channels (stage 0 is after patch_embed / stage 0 blocks).
        self.stage_adapters = nn.ModuleList([
            nn.Conv2d(embed_dim * (2 ** i), 3, kernel_size=1)
            for i in range(len(depths))
        ])

        if self.use_decoder:
            stage_configs = make_decoder_config(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                depths=depths,
                base_depth=decoder_depth,
                base_window_size=decoder_window_size,
            )
            self.decoder = SwinTransformerV2Decoder(
                stage_configs=stage_configs,
                window_size=decoder_window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=0.,
                attn_drop=0.,
                drop_path=drop_path_rate,
                act_layer='gelu',
                norm_layer=norm_layer,
                out_chans=out_chans,
            )
        else:
            self.decoder = None

    def get_exit_resolution(self, exit_stage=None):
        """Return spatial resolution (H, W) after the given exit_stage."""
        base = self.img_size // self.patch_size
        if exit_stage is None:
            exit_stage = len(self.depths) - 1
        res = base
        for i in range(1, exit_stage + 1):
            res //= 2
        return res, res

    def forward(self, x, decoder_pass=False, exit_stage=None):
        """
        Args:
            x: (B, C, H, W) input image
            decoder_pass: whether to also run the decoder
            exit_stage: encoder stage index to exit at (None = last stage)
        Returns:
            If decoder_pass=False: adapter output (B, 3, H_out, W_out)
            If decoder_pass=True: (adapter output, decoder output)
        """
        # Manual forward_features so we can exit early
        x = self.swin.patch_embed(x)
        if hasattr(self.swin, 'pos_drop'):
            x = self.swin.pos_drop(x)

        for i, layer in enumerate(self.swin.layers):
            x = layer(x)
            if exit_stage is not None and i == exit_stage:
                break

        if exit_stage is None:
            x = self.swin.norm(x)

        # (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)

        adapter_idx = exit_stage if exit_stage is not None else (len(self.swin.layers) - 1)
        x = self.stage_adapters[adapter_idx](x)

        if self.use_decoder and decoder_pass and self.decoder is not None:
            # Look up decoder start stage by matching output resolution
            out_res = x.shape[-1]
            start_stage = 0
            for i, cfg in enumerate(self.decoder.stage_configs):
                if cfg['in_res'] == out_res:
                    start_stage = i
                    break
            upsample_x = self.decoder(x, start_stage=start_stage)
            return x, upsample_x
        else:
            return x

    def set_stage_freezing(self, freeze_before_stage=0, exit_stage=None):
        """Freeze all encoder/decoder stages before the given index.

        freeze_before_stage=0  -> nothing frozen (train everything)
        freeze_before_stage=1  -> stage 0 encoder + adapter frozen; rest trainable
        freeze_before_stage=N  -> stages 0..N-1 frozen; stage N.. trainable

        Decoder stages are frozen based on the exit_stage resolution:
        decoder stages with input resolution > exit resolution are frozen
        (they were trained in earlier phases).
        """
        if exit_stage is None:
            exit_stage = freeze_before_stage
        exit_res, _ = self.get_exit_resolution(exit_stage)

        # Patch embed is tied to stage 0
        for p in self.swin.patch_embed.parameters():
            p.requires_grad = (freeze_before_stage <= 0)

        # Encoder layers + adapters
        for i, layer in enumerate(self.swin.layers):
            trainable = (i >= freeze_before_stage)
            for p in layer.parameters():
                p.requires_grad = trainable
            if i < len(self.stage_adapters):
                for p in self.stage_adapters[i].parameters():
                    p.requires_grad = trainable

        # Swin norm (only used when running full model)
        if hasattr(self.swin, 'norm'):
            for p in self.swin.norm.parameters():
                p.requires_grad = (freeze_before_stage >= len(self.swin.layers))

        # Decoder layers: stages with input resolution >= exit_res are on the
        # active reconstruction path and should be trainable. Stages with
        # input resolution < exit_res are not used in this phase.
        if self.use_decoder and self.decoder is not None:
            for i, stage in enumerate(self.decoder.stages):
                stage_in_res = self.decoder.stage_configs[i]['in_res']
                trainable = (stage_in_res >= exit_res)
                for p in stage.parameters():
                    p.requires_grad = trainable
            # Input adapters follow the same rule
            for i, adapter in enumerate(self.decoder.input_adapters):
                stage_in_res = self.decoder.stage_configs[i]['in_res']
                trainable = (stage_in_res >= exit_res)
                for p in adapter.parameters():
                    p.requires_grad = trainable
            # Norm and output adapter always trainable (lightweight)
            for p in self.decoder.norm.parameters():
                p.requires_grad = True
            for p in self.decoder.adapter_out.parameters():
                p.requires_grad = True


class FoundationModelWithSwin(nn.Module):
    """Foundation model (BiomedCLIP or MedCLIP) with optional Swin downsampling,
    Q-Former, and P-Tuning components.
    """
    def __init__(
        self,
        foundation_model,
        tokenizer,
        foundation_model_key="biomedclip",
        swin_pretrained_path=None,
        image_size=896,
        prompt_length=10,
        use_qformer=False,
        use_swin=False,
        use_ptuning=False,
        use_decoder=False,
        swin_patch_size=2,
        swin_embed_dim=48,
        swin_depths=None,
        swin_num_heads=None,
        swin_window_size=4,
        decoder_depth=2,
        decoder_window_size=4,
    ):
        super().__init__()
        # Store the foundation model
        self.foundation_model = foundation_model
        self.foundation_model_key = foundation_model_key

        # Get model config for embedding dimensions
        model_config = get_foundation_model_config(foundation_model_key)
        self.embed_dim = model_config["embed_dim"]
        self.foundation_image_size = model_config["image_size"]

        self.use_ptuning = use_ptuning
        if self.use_ptuning:
            self.ptuned_text_encoder = PEFTTextEncoder(
                base_text_encoder=self.foundation_model.text,
            )

        # Default Swin architecture for 896 -> 448 -> 224 -> 112
        if swin_depths is None:
            swin_depths = [2, 2, 2]
        if swin_num_heads is None:
            swin_num_heads = [2, 4, 8]

        self.use_swin = use_swin
        if self.use_swin:
            self.swin_with_adapter = SwinAutoencoderWithAdapter(
                img_size=image_size,
                patch_size=swin_patch_size,
                in_chans=1,
                embed_dim=swin_embed_dim,
                depths=swin_depths,
                num_heads=swin_num_heads,
                window_size=swin_window_size,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_rate=0,
                attn_drop_rate=0,
                drop_path_rate=0.1,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
                use_checkpoint=True,
                num_classes=0,
                use_decoder=use_decoder,
                out_chans=1,
                decoder_depth=decoder_depth,
                decoder_window_size=decoder_window_size,
            )
        else:
            self.swin_with_adapter = None

        if swin_pretrained_path:
            if self.swin_with_adapter:
                if os.path.isfile(swin_pretrained_path):
                    old_state_dict = {k: v.clone() for k, v in self.swin_with_adapter.state_dict().items()}
                    try:
                        state_dict = torch.load(swin_pretrained_path)
                        # Backward compatibility: old checkpoints used keys 'adapter', 'upsample_to_224', 'refine_224'
                        if 'stage_adapters' in state_dict:
                            # New format
                            self.swin_with_adapter.swin.load_state_dict(state_dict['swin'])
                            for i, sd in enumerate(state_dict['stage_adapters']):
                                self.swin_with_adapter.stage_adapters[i].load_state_dict(sd)
                            if self.swin_with_adapter.use_decoder and 'decoder' in state_dict:
                                self.swin_with_adapter.decoder.load_state_dict(state_dict['decoder'])
                        else:
                            # Old format: map single adapter -> stage_adapters[-1]
                            self.swin_with_adapter.swin.load_state_dict(state_dict['swin'])
                            if 'adapter' in state_dict:
                                last_idx = len(self.swin_with_adapter.stage_adapters) - 1
                                self.swin_with_adapter.stage_adapters[last_idx].load_state_dict(state_dict['adapter'])
                            if self.swin_with_adapter.use_decoder and 'decoder' in state_dict:
                                self.swin_with_adapter.decoder.load_state_dict(state_dict['decoder'])
                        if weights_changed(self.swin_with_adapter, old_state_dict):
                            print(f"Swin weights loaded and model parameters updated from {swin_pretrained_path}")
                        else:
                            print("Warning: Swin weights loaded, but model parameters did not change!")
                    except Exception as e:
                        print(f"Error loading Swin weights: {e}")
                else:
                    print(f"Pretrained path does not exist: {swin_pretrained_path}")
            else:
                print("No Swin model to load into")
        else:
            print("No path specified for Swin pretrained checkpoint.")

        self.learnable_clip_norm = LearnableClipNorm()

        self.use_qformer = use_qformer
        if use_qformer:
            # QFormer config adapts to foundation model embedding dimension
            qformer_kwargs = {
                "num_query_tokens": 64,
                "query_dim": 256,
                "encoder_dim": self.embed_dim,
                "num_layers": 6,
                "num_heads": 8
            }
            self.qformer = QFormer(**qformer_kwargs)

            self.qformer_head = nn.Linear(
                qformer_kwargs.get("query_dim", 256),
                self.embed_dim
            )

    @property
    def logit_scale(self):
        return self.foundation_model.logit_scale

    def encode_image(self, x, normalize=False):
        if self.swin_with_adapter is not None:
            x = self.swin_with_adapter(x)
            x = self.learnable_clip_norm(x)
        elif x.shape[2] != self.foundation_image_size or x.shape[3] != self.foundation_image_size:
            x = F.interpolate(x, size=(self.foundation_image_size, self.foundation_image_size), mode='bilinear', align_corners=False)

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = self.foundation_model.encode_image(x, normalize=False)

        if self.use_qformer:
            x = x.unsqueeze(1)
            x = self.qformer(x)
            x = x.mean(dim=1)
            x = self.qformer_head(x)

        if normalize:
            x = F.normalize(x, dim=-1)

        return x

    def swin_pass(self, x):
        x = self.swin_with_adapter(x)
        x = self.learnable_clip_norm(x)

        return x

    def encode_text(self, text, normalize=False):
        if self.use_ptuning:
            text_features = self.ptuned_text_encoder(text)
            if normalize:
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features
        return self.foundation_model.encode_text(text, normalize=normalize)

    def forward(self, batch_images, text_features, return_image_features=False):
        image_features = self.encode_image(batch_images, normalize=True)

        logit_scale = self.logit_scale.exp()

        logits = logit_scale * image_features @ text_features.t()

        if return_image_features:
            return logits, image_features

        return logits

    def freeze_swin_weights(self, requires_grad=False):
        if self.use_swin:
            for param in self.swin_with_adapter.parameters():
                param.requires_grad = requires_grad

    def freeze_swin_stages(self, freeze_before_stage=0, exit_stage=None):
        """Granular stage freezing for progressive training."""
        if self.use_swin:
            self.swin_with_adapter.set_stage_freezing(
                freeze_before_stage=freeze_before_stage,
                exit_stage=exit_stage,
            )

    def freeze_weights(self, current_epoch=0):
        if self.use_qformer:
            for param in self.qformer.parameters():
                param.requires_grad = True

        if self.use_ptuning:
            for param in self.ptuned_text_encoder.parameters():
                param.requires_grad = True

        if self.use_swin:
            for param in self.swin_with_adapter.parameters():
                param.requires_grad = True
            for param in self.learnable_clip_norm.parameters():
                param.requires_grad = False

        for param in self.foundation_model.parameters():
            param.requires_grad = False

        if (not self.use_swin) and (not self.use_qformer):
            for param in self.foundation_model.visual.parameters():
                param.requires_grad = True

        if (not self.use_ptuning):
            for param in self.foundation_model.text.parameters():
                param.requires_grad = True

    def print_parameter_count(self):
        def count_parameters(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        print("Parameter counts:")
        print(f"Foundation Model: {self.foundation_model_key}")
        print(f"Vision Encoder: {count_parameters(self.foundation_model.visual):,}")
        print(f"Text Encoder: {count_parameters(self.foundation_model.text):,}")
        if self.use_ptuning:
            print(f"P-Tuning: {count_parameters(self.ptuned_text_encoder):,}")
        if self.use_qformer:
            print(f"Q-Former: {count_parameters(self.qformer):,}")
            print(f"Q-Former Head: {count_parameters(self.qformer_head):,}")
        if self.use_swin:
            print(f"Swin Transformer: {count_parameters(self.swin_with_adapter.swin):,}")
            print(f"Swin Stage Adapters: {count_parameters(self.swin_with_adapter.stage_adapters):,}")
            if self.swin_with_adapter.use_decoder:
                print(f"Swin Decoder: {count_parameters(self.swin_with_adapter.decoder):,}")


def create_custom_preprocess(image_size, use_swin=False):
    if use_swin:
        return transforms.Compose([
            # transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])
    else:
        return transforms.Compose([
            # transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.CenterCrop(image_size),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])


def create_combined_model(
    tokenizer,
    foundation_model_key="biomedclip",
    swin_pretrained_path=None,
    image_size=896,
    use_qformer=False,
    use_swin=False,
    use_ptuning=False,
    use_decoder=False,
    swin_patch_size=2,
    swin_embed_dim=48,
    swin_depths=None,
    swin_num_heads=None,
    swin_window_size=4,
    decoder_depth=2,
    decoder_window_size=4,
    pretrained_path=None,  # deprecated, kept for compatibility
    model_name=None,  # deprecated, kept for compatibility
    num_classes=None,
):
    """Create combined model with specified foundation model.

    Args:
        tokenizer: Tokenizer for text encoding (deprecated, we get it from open_clip)
        foundation_model_key: 'biomedclip' or 'medclip'
        swin_pretrained_path: Path to pretrained Swin weights
        image_size: Input image size
        use_qformer: Whether to use Q-Former
        use_swin: Whether to use Swin downsampling
        use_ptuning: Whether to use P-Tuning
        use_decoder: Whether to use decoder
        swin_patch_size: Swin patch embed size (default 2 for 896)
        swin_embed_dim: Swin embed dimension (default 48)
        swin_depths: List of stage depths (default [2,2,2] for 896)
        swin_num_heads: List of stage heads (default [2,4,8] for 896)
        swin_window_size: Swin window size (default 4)
        decoder_depth: Decoder stage depth (default 2)
        decoder_window_size: Decoder window size (default 4)
        pretrained_path: Deprecated, kept for backward compatibility
        model_name: Deprecated, kept for backward compatibility

    Returns:
        combined_model, preprocess function
    """
    if foundation_model_key == "raddino":
        if use_swin or use_qformer or use_ptuning or use_decoder:
            raise ValueError("RAD-DINO baseline requires Swin, Q-Former, P-Tuning and decoder disabled")
        if num_classes is None:
            raise ValueError("RAD-DINO requires num_classes")
        from autoswin.raddino import create_raddino
        return create_raddino(num_classes)

    # Get foundation model configuration
    model_config = get_foundation_model_config(foundation_model_key)
    model_name_or_path = model_config["model_name"]
    
    print(f"Loading foundation model: {foundation_model_key} ({model_name_or_path})")
    
    # Load the foundation model and tokenizer from open_clip
    model, _, preprocess_clip = create_model_and_transforms(model_name_or_path, pretrained=None)
    tokenizer = get_tokenizer(model_name_or_path)

    combined_model = FoundationModelWithSwin(
        model,
        tokenizer,
        foundation_model_key=foundation_model_key,
        swin_pretrained_path=swin_pretrained_path,
        image_size=image_size,
        use_qformer=use_qformer,
        use_swin=use_swin,
        use_ptuning=use_ptuning,
        use_decoder=use_decoder,
        swin_patch_size=swin_patch_size,
        swin_embed_dim=swin_embed_dim,
        swin_depths=swin_depths,
        swin_num_heads=swin_num_heads,
        swin_window_size=swin_window_size,
        decoder_depth=decoder_depth,
        decoder_window_size=decoder_window_size,
    )

    # For Swin models, use custom preprocess; otherwise use model's preprocess
    if use_swin:
        preprocess = create_custom_preprocess(image_size=image_size, use_swin=True)
    else:
        # Use the model's built-in preprocess which handles the correct image size
        preprocess = preprocess_clip

    return combined_model, preprocess, tokenizer


def self_contained_create_combined_model(model_name, config_path):
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Get foundation model from config
    foundation_model = config.get("FOUNDATION_MODEL", "")
    if not foundation_model:
        raise ValueError("You must choose a foundation model ('medclip', 'biomedclip') in your config.")
    
    SWIN_HEAD = config["SWIN_HEAD"]
    USE_QFORMER = config["USE_QFORMER"]
    USE_PTUNING = config["USE_PTUNING"]
    IMAGE_SIZE = config["IMAGE_SIZE"]
    SWIN_PRETRAINED_PATH = config["SWIN_PRETRAINED_PATH"]

    combined_model, preprocess, tokenizer = create_combined_model(
        tokenizer=None,  # Will be created internally
        foundation_model_key=foundation_model,
        swin_pretrained_path=SWIN_PRETRAINED_PATH,
        image_size=IMAGE_SIZE,
        use_qformer=USE_QFORMER,
        use_swin=SWIN_HEAD,
        use_ptuning=USE_PTUNING,
        use_decoder=False,
    )

    return combined_model, preprocess, tokenizer

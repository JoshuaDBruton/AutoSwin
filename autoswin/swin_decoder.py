import math
from torch import nn
from timm.models.swin_transformer_v2 import SwinTransformerV2Block


class PatchExpand(nn.Module):
    def __init__(self, input_dim, expand_dim=None, upsample_rate=2):
        super().__init__()
        self.upsample_rate = upsample_rate
        self.expand_dim = expand_dim or input_dim // 2
        self.linear = nn.Linear(input_dim, self.expand_dim * (upsample_rate ** 2))
        self.pixel_shuffle = nn.PixelShuffle(upsample_rate)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.linear(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.pixel_shuffle(x)   # (B, C//4, H*2, W*2)
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H*2, W*2, C//4)
        return x


class SwinTransformerV2DecoderStage(nn.Module):
    def __init__(
        self,
        dim,
        out_dim,
        input_resolution,
        depth,
        upsample=False,
        num_heads=4,
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        proj_drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer='gelu',
        norm_layer=nn.LayerNorm,
        pretrained_window_size=0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerV2Block(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
            ) for i in range(depth)
        ])
        self.upsample = upsample
        if upsample:
            self.patch_expand = PatchExpand(dim, out_dim)

    def forward(self, x):
        # x: (B, H, W, C)
        for blk in self.blocks:
            x = blk(x)
        if self.upsample:
            x = self.patch_expand(x)
        return x


def make_decoder_config(img_size, patch_size, embed_dim, depths,
                        base_depth=2, base_window_size=4, mlp_ratio=4.0,
                        min_dim=12):
    """Build decoder stage configs from encoder architecture.

    The decoder mirrors the encoder in reverse: each upsample stage
    halves the channel count and doubles the spatial resolution.
    The last stage is a pass-through (no upsample) that feeds the
    final output adapter.
    """
    # Compute spatial resolution after each encoder stage
    base_res = img_size // patch_size
    stage_resolutions = [base_res]
    cur_res = base_res
    for i in range(len(depths)):
        if i > 0:
            cur_res //= 2
        stage_resolutions.append(cur_res)

    min_res = stage_resolutions[-1]
    num_upsamples = int(round(math.log2(img_size / min_res)))

    # Channel dimensions: start from encoder's last stage channels
    last_enc_dim = embed_dim * (2 ** (len(depths) - 1))
    dims = []
    d = last_enc_dim
    for i in range(num_upsamples + 1):
        dims.append(d)
        if i < num_upsamples:
            d = max(d // 2, min_dim)

    # Simple head heuristic: dim // 24, at least 1
    heads = [max(1, dim // 24) for dim in dims]

    stage_configs = []
    cur_res = min_res
    for i in range(num_upsamples + 1):
        next_res = cur_res * 2 if i < num_upsamples else cur_res
        stage_configs.append({
            'in_res': cur_res,
            'out_res': next_res,
            'dim': dims[i],
            'depth': base_depth,
            'heads': heads[i],
            'upsample': i < num_upsamples,
        })
        cur_res = next_res

    return stage_configs


class SwinTransformerV2Decoder(nn.Module):
    def __init__(
        self,
        stage_configs,
        window_size=4,
        mlp_ratio=4.,
        qkv_bias=True,
        proj_drop=0.,
        attn_drop=0.,
        drop_path=0.1,
        act_layer='gelu',
        norm_layer=nn.LayerNorm,
        pretrained_window_sizes=None,
        out_chans=1,
    ):
        super().__init__()
        self.stage_configs = stage_configs
        self.num_stages = len(stage_configs)

        # One input adapter per possible entry stage (3 -> stage_dim)
        self.input_adapters = nn.ModuleList([
            nn.Conv2d(3, cfg['dim'], kernel_size=1)
            for cfg in stage_configs
        ])

        if pretrained_window_sizes is None:
            pretrained_window_sizes = [0] * self.num_stages

        self.stages = nn.ModuleList()
        for i, cfg in enumerate(stage_configs):
            out_dim = stage_configs[i + 1]['dim'] if i < self.num_stages - 1 else cfg['dim']
            self.stages.append(
                SwinTransformerV2DecoderStage(
                    dim=cfg['dim'],
                    out_dim=out_dim,
                    input_resolution=(cfg['in_res'], cfg['in_res']),
                    depth=cfg['depth'],
                    upsample=cfg['upsample'],
                    num_heads=cfg['heads'],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    pretrained_window_size=pretrained_window_sizes[i],
                )
            )

        self.norm = norm_layer(stage_configs[-1]['dim'])
        self.adapter_out = nn.Conv2d(stage_configs[-1]['dim'], out_chans, kernel_size=1)

    def forward(self, x, start_stage=0):
        """
        Args:
            x: (B, 3, H, W) where H,W matches stage_configs[start_stage]['in_res']
            start_stage: which decoder stage to start from (0 = full path)
        Returns:
            (B, out_chans, target_H, target_W)
        """
        x = self.input_adapters[start_stage](x)  # (B, dim, H, W)
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)

        for i in range(start_stage, self.num_stages):
            x = self.stages[i](x)

        x = self.norm(x)
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        x = self.adapter_out(x)
        return x

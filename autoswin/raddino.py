"""RAD-DINO CLS classifier, following ../rad-dino/run.py's baseline design."""
import torch
import numpy as np
from PIL import Image
from torch import nn
from torchvision import transforms as T
from transformers import AutoImageProcessor, AutoModel

MODEL_ID = "microsoft/rad-dino"
REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"


def display_rgb(image):
    """Match the sibling baseline's signed-HDF5 intensity recovery."""
    pixels = np.asarray(image, dtype=np.float32)
    low, high = float(pixels.min()), float(pixels.max())
    if not np.isfinite(pixels).all() or high <= low:
        raise ValueError("RAD-DINO expects a finite, nonconstant image")
    pixels = np.rint((pixels - low) / (high - low) * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(pixels).convert("RGB")


class RadDinoClassifier(nn.Module):
    def __init__(self, num_classes, encoder=None):
        super().__init__()
        self.encoder = encoder if encoder is not None else AutoModel.from_pretrained(
            MODEL_ID, revision=REVISION, attn_implementation="sdpa")
        self.encoder.gradient_checkpointing_enable()
        self.head = nn.Linear(self.encoder.config.hidden_size, num_classes)
        self.swin_with_adapter = None

    def forward(self, images, text_features=None, return_image_features=False):
        features = self.encoder(pixel_values=images).last_hidden_state[:, 0]
        logits = self.head(features)
        return (logits, features) if return_image_features else logits

    def freeze_weights(self):
        self.requires_grad_(True)

    def print_parameter_count(self):
        print(f"RAD-DINO trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")


def create_raddino(num_classes):
    proc = AutoImageProcessor.from_pretrained(MODEL_ID, revision=REVISION, use_fast=False)
    preprocess = T.Compose([
        T.Lambda(display_rgb),
        T.Resize(proc.size["shortest_edge"], interpolation=T.InterpolationMode.LANCZOS),
        T.CenterCrop((proc.crop_size["height"], proc.crop_size["width"])),
        T.ToTensor(), T.Normalize(proc.image_mean, proc.image_std),
    ])
    return RadDinoClassifier(num_classes), preprocess, None

"""Synthetic pretrained-model training step; no patient data or WandB required."""
import argparse
import numpy as np
import torch
from PIL import Image
from autoswin.models import create_combined_model
from autoswin.mccv import seed_fold
from autoswin.multilabel_metrics import macro_auc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["biomedclip", "raddino"], default="biomedclip")
    args = parser.parse_args()
    for fold in range(5):
        assert seed_fold(fold) == 2026 + fold
        first = torch.rand(3)
        seed_fold(fold)
        assert torch.equal(first, torch.rand(3))
    assert macro_auc(np.array([[0, 1], [1, 0]]), np.array([[.1, .8], [.9, .2]])) == 1
    seed_fold(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess, tokenizer = create_combined_model(None, foundation_model_key=args.model, num_classes=2)
    model.to(device).train()
    model.freeze_weights()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-5)
    image = Image.fromarray(np.random.randint(0, 256, (512, 512), dtype=np.uint8))
    images = preprocess(image).unsqueeze(0).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        text = model.encode_text(tokenizer(["silicosis", "tuberculosis"], context_length=256).to(device), normalize=True) if tokenizer else None
        logits = model(images, text)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([[1., 0.]], device=device))
    assert torch.isfinite(loss)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    assert logits.shape == (1, 2)
    print(f"PASS {args.model}: forward/backward/optimizer, loss={loss.item():.4f}")
    if device.type == "cuda":
        print(f"Peak GPU allocation: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()

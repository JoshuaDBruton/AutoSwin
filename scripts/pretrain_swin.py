"""
Pretrain the Swin adapter (downsampling from high-res to intermediate resolutions).

Supports progressive layer-wise training via SWIN_PHASE_EPOCHS.
Example config:
    SWIN_PHASE_EPOCHS: [50, 50, 50]
    # Phase 0: 50 epochs at exit_stage=0 (freeze_before=0)
    # Phase 1: 50 epochs at exit_stage=1 (freeze_before=1)
    # Phase 2: 50 epochs at exit_stage=2 (freeze_before=2)

If SWIN_PHASE_EPOCHS is empty / not set, uses SWIN_EXIT_STAGE directly.

Usage:
    export BIOMED_DATA_DIR=/home/joshua/data
    python pretrain_swin.py --config configs/pretrain_swin_config.yaml
    python pretrain_swin.py --config configs/pretrain_swin_config.yaml --no-wandb
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from autoswin.config import load_config, validate_data_files
from autoswin.dataloader import get_dataloaders
from autoswin.experiment import Experiment, WandbLogger
from autoswin.models import create_combined_model
from autoswin.utils import plot_random_images


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain Swin adapter")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to experiment config YAML")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    return parser.parse_args()


def build_phase_schedule(cfg):
    """Return list of (exit_stage, freeze_before, epochs) tuples."""
    if getattr(cfg, 'swin_phase_epochs', None) and len(cfg.swin_phase_epochs) > 0:
        phases = []
        for i, epochs in enumerate(cfg.swin_phase_epochs):
            phases.append({
                'exit_stage': i,
                'freeze_before': i,
                'epochs': epochs,
            })
        return phases
    # Single-phase mode
    exit_stage = cfg.swin_exit_stage if cfg.swin_exit_stage >= 0 else (len(cfg.swin_depths) - 1)
    return [{
        'exit_stage': exit_stage,
        'freeze_before': 0,
        'epochs': cfg.epochs,
    }]


def save_swin_checkpoint(swin_model, exp, tag):
    ckpt = {
        'swin': swin_model.swin.state_dict(),
        'stage_adapters': [adapter.state_dict() for adapter in swin_model.stage_adapters],
    }
    if swin_model.use_decoder and swin_model.decoder is not None:
        ckpt['decoder'] = swin_model.decoder.state_dict()
    exp.save_checkpoint(ckpt, tag=tag)


def load_swin_checkpoint(swin_model, ckpt_path):
    state_dict = torch.load(ckpt_path)
    swin_model.swin.load_state_dict(state_dict['swin'])
    if 'stage_adapters' in state_dict:
        for i, sd in enumerate(state_dict['stage_adapters']):
            swin_model.stage_adapters[i].load_state_dict(sd)
    elif 'adapter' in state_dict:
        # Old-format backward compatibility: load into last adapter
        last_idx = len(swin_model.stage_adapters) - 1
        swin_model.stage_adapters[last_idx].load_state_dict(state_dict['adapter'])
    if swin_model.use_decoder and swin_model.decoder is not None and 'decoder' in state_dict:
        swin_model.decoder.load_state_dict(state_dict['decoder'])


def train_phase(swin_model, swindp_model, train_loader, val_loader, device,
                optimizer, criterion, cfg, exp, wb, phase_info,
                global_epoch_start=0):
    """Train a single phase and return the next global epoch index."""
    exit_stage = phase_info['exit_stage']
    freeze_before = phase_info['freeze_before']
    epochs = phase_info['epochs']

    swin_model.set_stage_freezing(freeze_before_stage=freeze_before)

    # Print trainable params for this phase
    print(f"\n=== Phase: exit_stage={exit_stage}, freeze_before={freeze_before}, epochs={epochs} ===")
    for name, param in swin_model.named_parameters():
        if param.requires_grad:
            print(f"  [TRAIN] {name}")

    swin_model.train()
    global_epoch = global_epoch_start

    for epoch in range(epochs):
        epoch_loss = 0
        total = 0

        for batch_images, batch_labels in tqdm(train_loader, desc=f"Phase {exit_stage} Epoch {epoch}"):
            batch_images = batch_images.to(device)

            if cfg.use_decoder:
                ds, recon = swindp_model(batch_images, decoder_pass=True, exit_stage=exit_stage)
                recon_loss = criterion(recon, batch_images)
            else:
                ds = swindp_model(batch_images, exit_stage=exit_stage)

            # Fidelity target at the actual output resolution
            target_size = ds.shape[-2:]
            target_images = F.interpolate(
                batch_images, size=target_size, mode='bilinear', align_corners=False
            )
            if target_images.shape[1] == 1:
                target_images = target_images.repeat(1, 3, 1, 1)

            fidelity_loss = criterion(ds, target_images)

            if cfg.use_decoder:
                loss = (cfg.fidelity_weight * fidelity_loss) + ((1 - cfg.fidelity_weight) * recon_loss)
            else:
                loss = fidelity_loss

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += 1

        epoch_loss_avg = epoch_loss / total if total > 0 else 0
        print(f"Phase {exit_stage} Epoch {epoch} Loss: {epoch_loss_avg}")

        swin_model.eval()
        fig = plot_random_images(val_loader, swin_model, device)
        wb.log({"epoch_loss": epoch_loss_avg}, step=global_epoch)
        wb.log_image("swin_downsamples", fig, step=global_epoch)
        exp.log_metrics(epoch=global_epoch, epoch_loss=epoch_loss_avg, phase=exit_stage)
        exp.save_figure(fig, f"swin_downsamples_phase{exit_stage}", epoch=global_epoch)
        plt.close(fig)

        swin_model.train()
        global_epoch += 1

    return global_epoch


def main():
    args = parse_args()
    cfg = load_config(args.config)

    errors = validate_data_files(cfg)
    if errors:
        print("Missing data files:", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)

    if cfg.fidelity_weight < 1 and not cfg.use_decoder:
        print("ERROR: Fidelity weight < 1 requires USE_DECODER=True", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Experiment tracking ---
    exp = Experiment.create(cfg, base_dir="experiments", name_suffix="pretrain-swin")
    wb = WandbLogger(enabled=not args.no_wandb)

    print(f"Experiment directory: {exp.dir}")

    # --- Resume from checkpoint if available ---
    latest = None
    for f in glob.glob(str(exp.checkpoints_dir / "latest-*.pth")):
        latest = f

    latest_epoch = -1
    swin_pretrained_path = None
    phase_start = 0
    if latest:
        swin_pretrained_path = latest
        basename = os.path.basename(latest)
        latest_epoch = int(basename.replace("latest-", "").replace(".pth", ""))
        print(f"Found {latest}... starting from epoch {latest_epoch + 1}...")

    # --- Model ---
    swin_model, preprocess, tokenizer = create_combined_model(
        tokenizer=None,
        foundation_model_key=cfg.foundation_model,
        swin_pretrained_path=swin_pretrained_path,
        image_size=cfg.image_size,
        use_qformer=cfg.use_qformer,
        use_swin=cfg.swin_head,
        use_ptuning=cfg.use_ptuning,
        use_decoder=cfg.use_decoder,
        swin_patch_size=cfg.swin_patch_size,
        swin_embed_dim=cfg.swin_embed_dim,
        swin_depths=cfg.swin_depths,
        swin_num_heads=cfg.swin_num_heads,
        swin_window_size=cfg.swin_window_size,
        decoder_depth=cfg.decoder_depth,
        decoder_window_size=cfg.decoder_window_size,
    )

    swin_model = swin_model.swin_with_adapter

    if torch.cuda.device_count() > 1:
        swindp_model = torch.nn.DataParallel(swin_model)
    else:
        swindp_model = swin_model

    swin_model.to(device)

    # Try to infer resume phase from checkpoint metadata if available
    resume_phase = 0
    if latest:
        try:
            ckpt_meta = torch.load(latest)
            resume_phase = ckpt_meta.get('phase', 0)
        except Exception:
            resume_phase = 0

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, swin_model.parameters()),
        lr=cfg.learning_rate,
        weight_decay=1e-4,
    )

    criterion = nn.MSELoss()

    # --- Data ---
    train_loader, val_loader, _ = get_dataloaders(
        cfg.resolved_data_path,
        preprocess,
        train_split=0.8,
        batch_size=cfg.batch_size,
        labels_key="lab",
        split_file=cfg.resolved_split_file,
        augmentations=None,
        oversample=False,
    )

    # --- WandB ---
    wb.init(
        project=cfg.project_name,
        name=f"pretrain-swin-{exp.dir.name}",
        config={
            "loss_function": "MSELoss",
            "learning_rate": cfg.learning_rate,
            "dataset": cfg.resolved_data_path,
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "use_decoder": cfg.use_decoder,
            "fidelity_weight": cfg.fidelity_weight,
            "swin_depths": cfg.swin_depths,
            "swin_num_heads": cfg.swin_num_heads,
            "swin_patch_size": cfg.swin_patch_size,
        },
    )

    # Initial visualisation
    swin_model.eval()
    fig = plot_random_images(val_loader, swin_model, device)
    wb.log_image("initial_swin_downsamples", fig)
    exp.save_figure(fig, "initial_swin_downsamples")
    plt.close(fig)

    swin_model.train()

    # --- Training loop ---
    phase_schedule = build_phase_schedule(cfg)
    global_epoch = latest_epoch + 1

    for phase_idx, phase_info in enumerate(phase_schedule):
        if phase_idx < resume_phase:
            print(f"Skipping phase {phase_idx} (already completed in resume)")
            continue

        global_epoch = train_phase(
            swin_model, swindp_model, train_loader, val_loader, device,
            optimizer, criterion, cfg, exp, wb, phase_info,
            global_epoch_start=global_epoch,
        )

        # Save phase checkpoint
        save_swin_checkpoint(swin_model, exp, tag=f"phase{phase_idx}-latest-{global_epoch - 1}")
        exp.remove_checkpoints("latest")
        save_swin_checkpoint(swin_model, exp, tag=f"latest-{global_epoch - 1}")

    wb.finish()
    print("Swin pretraining complete.")


if __name__ == "__main__":
    main()

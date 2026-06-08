"""
Run a single fold with the torchxrayvision ResNet model (resolution experiment).

Usage:
    export BIOMED_DATA_DIR=/home/joshua/data
    python run_fold_resolution.py --fold_number 0 --image_size 224 --config configs/resolution_config.yaml
"""

import argparse
import sys

import torch

from autoswin.config import load_config, validate_data_files
from autoswin.experiment import Experiment, WandbLogger
from autoswin.mccv import MonteCarloCrossValidator
from autoswin.txrv_wrapper import init_torchxrayvision_resnet_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run single fold with torchxrayvision ResNet")
    parser.add_argument("--image_size", type=int, required=True, help="Image size")
    parser.add_argument("--fold_number", type=int, required=True, help="Fold number")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to experiment config YAML")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    return parser.parse_args()


def main():
    args = parse_args()

    # Override image_size from CLI
    cfg = load_config(args.config, overrides={"image_size": args.image_size})

    # Sanity check: ResNet doesn't support Swin/QFormer/PTuning
    if cfg.swin_head or cfg.use_qformer or cfg.use_ptuning:
        print("ERROR: Swin, Q-Former, and P-Tuning are incompatible with the txrv ResNet model.", file=sys.stderr)
        sys.exit(1)

    errors = validate_data_files(cfg)
    if errors:
        print("Missing data files:", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = len(cfg.label_descriptions)

    def resnet_init_fn():
        return init_torchxrayvision_resnet_model(num_classes=num_classes)

    if cfg.use_ordinal_class or cfg.use_multilabel:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    exp = Experiment.create(cfg, base_dir="experiments", name_suffix=f"resnet-{args.image_size}-fold{args.fold_number}")
    wb = WandbLogger(enabled=not args.no_wandb)

    print(f"Experiment directory: {exp.dir}")
    print(f"Image size: {args.image_size}, Fold: {args.fold_number}")

    mccv = MonteCarloCrossValidator(
        cfg=cfg,
        model_init_func=resnet_init_fn,
        tokenizer=None,
        loss_fn=loss_fn,
        device=device,
        experiment=exp,
        wandb_logger=wb,
    )

    group_name = (cfg.group_name or cfg.project_name).format(suffix=args.image_size)
    mccv.run_fold(args.fold_number, group_name=group_name)


if __name__ == "__main__":
    main()

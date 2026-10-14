"""
Run a single fold of MCCV. Used for distributed / HPC execution.

Usage:
    export BIOMED_DATA_DIR=/home/joshua/data
    python run_fold.py --fold_number 0 --config configs/swin_train_config.yaml
    python run_fold.py --fold_number 0 --config configs/swin_train_config.yaml --no-wandb
"""

import argparse
import sys

import torch

from autoswin.config import load_config, validate_data_files
from autoswin.experiment import Experiment, WandbLogger
from autoswin.mccv import MonteCarloCrossValidator
from autoswin.models import create_combined_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single MCCV fold")
    parser.add_argument("--fold_number", type=int, required=True, help="Fold number")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to experiment config YAML")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config and set fold-specific checkpoint directory
    cfg = load_config(args.config)
    
    # Set fold-specific checkpoint directory to avoid conflicts
    cfg.resolved_checkpoint_dir = f"{cfg.checkpoint_save_dir}-fold{args.fold_number}".format(
        project_name=cfg.project_name,
        group_name=cfg.group_name,
    )

    # Validate data files
    errors = validate_data_files(cfg)
    if errors:
        print("Missing data files:", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def model_init_fn():
        model, preprocess, _ = create_combined_model(
            tokenizer=None,
            num_classes=len(cfg.label_descriptions) + int(cfg.use_ordinal_class),
            foundation_model_key=cfg.foundation_model,
            swin_pretrained_path=cfg.swin_pretrained_path,
            image_size=cfg.image_size,
            use_qformer=cfg.use_qformer,
            use_swin=cfg.swin_head,
            use_ptuning=cfg.use_ptuning,
            use_decoder=cfg.use_decoder,
        )
        return model, preprocess

    if cfg.use_ordinal_class or cfg.use_multilabel:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    # Initialize model once to get the tokenizer
    _, _, tokenizer = create_combined_model(
        tokenizer=None,
        num_classes=len(cfg.label_descriptions) + int(cfg.use_ordinal_class),
        foundation_model_key=cfg.foundation_model,
        swin_pretrained_path=cfg.swin_pretrained_path,
        image_size=cfg.image_size,
        use_qformer=cfg.use_qformer,
        use_swin=cfg.swin_head,
        use_ptuning=cfg.use_ptuning,
        use_decoder=cfg.use_decoder,
    )

    # --- Experiment tracking ---
    # Update project name to include foundation model
    original_project_name = cfg.project_name
    cfg.project_name = f"{cfg.foundation_model}-{original_project_name}"
    
    exp = Experiment.create(cfg, base_dir="experiments", name_suffix=f"fold{args.fold_number}")
    wb = WandbLogger(enabled=not args.no_wandb)

    print(f"Experiment directory: {exp.dir}")
    print(f"Fold: {args.fold_number}")
    print(f"Foundation Model: {cfg.foundation_model}")
    print(f"Config: {args.config}")

    # --- Run ---
    mccv = MonteCarloCrossValidator(
        cfg=cfg,
        model_init_func=model_init_fn,
        tokenizer=tokenizer,
        loss_fn=loss_fn,
        device=device,
        experiment=exp,
        wandb_logger=wb,
    )

    group_name = cfg.group_name or cfg.project_name
    mccv.run_fold(args.fold_number, group_name=group_name)


if __name__ == "__main__":
    main()

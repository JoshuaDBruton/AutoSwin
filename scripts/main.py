"""
Main entry point: runs full k-fold Monte-Carlo cross validation.

Usage:
    export BIOMED_DATA_DIR=/home/joshua/data
    python main.py --config configs/swin_train_config.yaml
    python main.py --config configs/swin_train_config.yaml --no-wandb
"""

import argparse
import sys

import torch

from autoswin.config import ExperimentConfig, load_config, validate_data_files
from autoswin.experiment import Experiment, WandbLogger
from autoswin.mccv import MonteCarloCrossValidator
from autoswin.models import create_combined_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run k-fold MCCV experiment")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to experiment config YAML")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Validate data files exist before starting
    errors = validate_data_files(cfg)
    if errors:
        print("Missing data files:", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Foundation model is now configured via config
    # cfg.foundation_model contains 'biomedclip' or 'medclip'
    
    def model_init_fn():
        model, preprocess, _ = create_combined_model(
            tokenizer=None,  # Will be created internally based on foundation model
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

    # --- Experiment tracking ---
    # Update project name to include foundation model
    original_project_name = cfg.project_name
    cfg.project_name = f"{cfg.foundation_model}-{original_project_name}"
    
    exp = Experiment.create(cfg, base_dir="experiments")
    wb = WandbLogger(enabled=not args.no_wandb)

    print(f"Experiment directory: {exp.dir}")
    print(f"WandB: {'enabled' if wb.enabled else 'disabled'}")
    print(f"Config: {args.config}")
    print(f"Foundation Model: {cfg.foundation_model}")
    print(f"Data: {cfg.resolved_data_path}")
    
    # Initialize model once to get the tokenizer
    model, preprocess, tokenizer = model_init_fn()

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

    mccv.run_k_iterations(cfg.num_folds)


if __name__ == "__main__":
    main()

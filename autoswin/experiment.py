"""
Experiment tracking and results management.

Handles:
- Creating experiment directories with unique IDs
- Saving config snapshots and git state
- Local JSON results logging (WandB-independent)
- Optional WandB integration
"""

from __future__ import annotations

import json
import os
import subprocess
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from autoswin.config import ExperimentConfig


# ---------------------------------------------------------------------------
# Eval result dataclass — replaces the 11-element tuples from evaluate()
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Structured result from a single evaluation pass."""
    accuracy: float = 0.0
    sensitivity: float = 0.0
    specificity: float = 0.0
    f1: float = 0.0
    bin_accuracy: Optional[float] = None
    bin_sensitivity: Optional[float] = None
    bin_specificity: Optional[float] = None
    bin_f1: Optional[float] = None
    average_auc: Optional[float] = None
    ml_metrics_dict: Optional[dict] = None
    # confusion matrix figure — not serialised, used for logging only
    confusion_fig: Any = None

    def to_dict(self) -> dict:
        """Serialisable dictionary (excludes matplotlib figures)."""
        d = asdict(self)
        d.pop("confusion_fig", None)
        return d


# ---------------------------------------------------------------------------
# Git state capture
# ---------------------------------------------------------------------------

def _get_git_info() -> dict[str, str]:
    """Capture current git commit and dirty status."""
    info = {"commit": "unknown", "dirty": False, "branch": "unknown"}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info["dirty"] = len(status) > 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return info


# ---------------------------------------------------------------------------
# Experiment directory manager
# ---------------------------------------------------------------------------

class Experiment:
    """Manages a single experiment's directory, config snapshot, and results.

    Usage:
        exp = Experiment.create(cfg, base_dir="experiments")
        exp.log_metrics(epoch=1, train_loss=0.5, val_acc=0.82)
        exp.log_eval("val", epoch=5, result=eval_result)
        exp.save_checkpoint(model.state_dict(), tag="best")
    """

    def __init__(self, experiment_dir: Path, cfg: ExperimentConfig):
        self.dir = experiment_dir
        self.cfg = cfg
        self.metrics_path = self.dir / "metrics.jsonl"
        self.checkpoints_dir = self.dir / "checkpoints"
        self.figures_dir = self.dir / "figures"

    @classmethod
    def create(
        cls,
        cfg: ExperimentConfig,
        base_dir: str = "experiments",
        name_suffix: str = "",
    ) -> "Experiment":
        """Create a new experiment directory with config snapshot."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        short_name = cfg.project_name
        if cfg.group_name:
            short_name += f"-{cfg.group_name}"
        if name_suffix:
            short_name += f"-{name_suffix}"

        dir_name = f"{timestamp}-{short_name}"
        exp_dir = Path(base_dir) / dir_name

        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "checkpoints").mkdir(exist_ok=True)
        (exp_dir / "figures").mkdir(exist_ok=True)

        exp = cls(exp_dir, cfg)

        # Save config snapshot
        with open(exp_dir / "config.yaml", "w") as f:
            yaml.dump(cfg.to_dict(), f, default_flow_style=False, sort_keys=False)

        # Save git info
        git_info = _get_git_info()
        with open(exp_dir / "git_info.json", "w") as f:
            json.dump(git_info, f, indent=2)

        return exp

    @classmethod
    def load(cls, experiment_dir: str) -> "Experiment":
        """Load an existing experiment from its directory."""
        exp_dir = Path(experiment_dir)
        config_path = exp_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"No config.yaml found in {exp_dir}")

        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)

        cfg = ExperimentConfig(**{
            k: v for k, v in raw.items()
            if k in ExperimentConfig.__dataclass_fields__
        })
        return cls(exp_dir, cfg)

    # --- Metrics logging ---

    def log_metrics(self, epoch: int, fold: Optional[int] = None, **metrics):
        """Append a metrics record to the JSONL log.

        Each line is a self-contained JSON object with epoch, fold,
        timestamp, and all metric key/value pairs.
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
        }
        if fold is not None:
            record["fold"] = fold
        record.update(metrics)

        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record, cls=_NumpyEncoder) + "\n")

    def log_eval(self, prefix: str, epoch: int, result: EvalResult, fold: Optional[int] = None):
        """Log an EvalResult with a prefix (e.g. 'train', 'val', 'test')."""
        metrics = {f"{prefix}_{k}": v for k, v in result.to_dict().items() if v is not None}
        self.log_metrics(epoch=epoch, fold=fold, **metrics)

    def log_fold_summary(self, fold: int, test_result: EvalResult):
        """Log the final test result for a completed fold."""
        record = {
            "type": "fold_summary",
            "timestamp": datetime.now().isoformat(),
            "fold": fold,
        }
        record.update({f"test_{k}": v for k, v in test_result.to_dict().items() if v is not None})

        with open(self.dir / "fold_summaries.jsonl", "a") as f:
            f.write(json.dumps(record, cls=_NumpyEncoder) + "\n")

    # --- Checkpoints ---

    def save_checkpoint(self, state_dict, tag: str = "latest"):
        """Save a model checkpoint. Overwrites previous checkpoint with same tag."""
        path = self.checkpoints_dir / f"{tag}.pth"
        import torch
        torch.save(state_dict, path)
        return path

    def remove_checkpoints(self, tag_prefix: str):
        """Remove all checkpoints matching a prefix."""
        for p in self.checkpoints_dir.glob(f"{tag_prefix}*"):
            p.unlink()

    # --- Figures ---

    def save_figure(self, fig, name: str, epoch: Optional[int] = None):
        """Save a matplotlib figure to the figures directory."""
        if epoch is not None:
            name = f"{name}_epoch{epoch}"
        path = self.figures_dir / f"{name}.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        return path

    # --- Reading results back ---

    def load_metrics(self) -> list[dict]:
        """Load all metric records from the JSONL log."""
        records = []
        if self.metrics_path.exists():
            with open(self.metrics_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        return records

    def load_fold_summaries(self) -> list[dict]:
        """Load fold summary records."""
        path = self.dir / "fold_summaries.jsonl"
        records = []
        if path.exists():
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        return records


# ---------------------------------------------------------------------------
# Optional WandB wrapper
# ---------------------------------------------------------------------------

class WandbLogger:
    """Thin wrapper around wandb that can be disabled.

    Usage:
        logger = WandbLogger(enabled=True)
        logger.init(project="my-project", name="run-1", config={...})
        logger.log({"loss": 0.5}, step=1)
        logger.finish()
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._run = None

    def init(self, **kwargs):
        if not self.enabled:
            return
        import wandb
        wandb.login()
        self._run = wandb.init(**kwargs)

    def log(self, data: dict, step: Optional[int] = None):
        if not self.enabled:
            return
        import wandb
        wandb.log(data, step=step)

    def log_image(self, key: str, fig, step: Optional[int] = None):
        """Log a matplotlib figure as an image."""
        if not self.enabled:
            return
        import wandb
        wandb.log({key: wandb.Image(fig)}, step=step)

    def finish(self):
        if not self.enabled:
            return
        import wandb
        wandb.finish()
        self._run = None


# ---------------------------------------------------------------------------
# JSON encoder for numpy types
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

"""
Unified configuration system.

All experiment config flows through the ExperimentConfig dataclass.
Entry points call `load_config(path)` to get a validated config object
instead of manually extracting YAML keys.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Label descriptions live in a separate file so configs stay short.
# ---------------------------------------------------------------------------

_LABELS_FILE = Path(__file__).parent / "labels.yaml"


def load_label_descriptions() -> dict[str, list[str]]:
    """Load all label-set definitions from labels.yaml."""
    if not _LABELS_FILE.exists():
        raise FileNotFoundError(
            f"Label descriptions file not found: {_LABELS_FILE}\n"
            "Run the migration or create labels.yaml from config.yaml's LABEL_DESCRIPTIONS block."
        )
    with open(_LABELS_FILE, "r") as f:
        data = yaml.safe_load(f)
    return data


# ---------------------------------------------------------------------------
# Data root
# ---------------------------------------------------------------------------

def get_data_root() -> Path:
    """Return the data root directory from BIOMED_DATA_DIR env var.

    Fails loudly if the variable is not set so you never silently
    train against the wrong directory on a new machine.
    """
    raw = os.environ.get("BIOMED_DATA_DIR")
    if raw is None:
        print(
            "ERROR: Environment variable BIOMED_DATA_DIR is not set.\n"
            "Set it to the directory containing your .h5 and .json data files.\n"
            "  export BIOMED_DATA_DIR=/home/joshua/data",
            file=sys.stderr,
        )
        sys.exit(1)
    p = Path(raw)
    if not p.is_dir():
        print(
            f"ERROR: BIOMED_DATA_DIR points to a path that does not exist: {p}",
            file=sys.stderr,
        )
        sys.exit(1)
    return p


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Complete experiment configuration.

    Every field has a sensible default so that experiment YAML files
    only need to specify what they *change*.
    """

    # --- Identity / tracking ---
    project_name: str = "biomedclip"
    group_name: str = ""

    # --- Architecture toggles ---
    foundation_model: str = ""  # 'biomedclip' or 'medclip' - REQUIRED
    swin_head: bool = False
    use_qformer: bool = False
    use_ptuning: bool = False
    use_ordinal_class: bool = False
    use_multilabel: bool = True
    use_autocast: bool = False
    use_decoder: bool = False

    # --- Swin architecture (new progressive training support) ---
    swin_patch_size: int = 2
    swin_embed_dim: int = 48
    swin_depths: list = field(default_factory=lambda: [2, 2, 2])
    swin_num_heads: list = field(default_factory=lambda: [2, 4, 8])
    swin_window_size: int = 4
    decoder_depth: int = 2
    decoder_window_size: int = 4
    swin_exit_stage: int = -1  # -1 = use all stages; 0 = exit after stage 0, etc.
    swin_phase_epochs: list = field(default_factory=list)  # Progressive phase epochs; empty = disabled

    # --- Data ---
    labels_key: str = "lab"
    label_set: str = "VINDR_BASIC"
    image_size: int = 224
    # Paths relative to BIOMED_DATA_DIR.  Use {image_size} as placeholder.
    data_path: str = ""
    test_data_path: str = ""
    split_file: str = ""

    # --- Training ---
    batch_size: int = 2
    learning_rate: float = 5e-5
    epochs: int = 150
    num_folds: int = 10
    oversample: bool = False
    alignment_loss_weight: float = 0.0
    fidelity_weight: float = -1.0

    # --- Scheduling ---
    foundation_model_frozen_epochs: int = 0
    swin_frozen_epochs: int = 0
    learning_rate_decay_step_size: int = 25
    learning_rate_decay_gamma: float = 0.5

    # --- Checkpointing ---
    checkpoint_save_dir: str = "checkpoints/{project_name}-{group_name}"
    swin_pretrained_path: Optional[str] = None

    # --- Derived (populated after load) ---
    label_descriptions: list[str] = field(default_factory=list)
    resolved_data_path: str = ""
    resolved_test_data_path: str = ""
    resolved_split_file: str = ""
    resolved_checkpoint_dir: str = ""

    def __post_init__(self):
        if self.use_ordinal_class and self.use_multilabel:
            raise ValueError("use_ordinal_class and use_multilabel are incompatible.")
        
        # Validate foundation model
        valid_models = ["biomedclip", "medclip"]
        if not self.foundation_model:
            raise ValueError(
                "You must choose a foundation model ('medclip', 'biomedclip') in your config."
            )
        if self.foundation_model not in valid_models:
            raise ValueError(
                f"Invalid foundation_model '{self.foundation_model}'. "
                f"Must be one of: {', '.join(valid_models)}"
            )

    @property
    def num_classes(self) -> int:
        if self.use_ordinal_class:
            return len(self.label_descriptions) + 1
        return len(self.label_descriptions)

    def to_dict(self) -> dict:
        """Serialisable dictionary for saving alongside experiments."""
        return asdict(self)


# ---------------------------------------------------------------------------
# YAML key mapping
# ---------------------------------------------------------------------------
# Maps YAML keys (UPPER_SNAKE) to dataclass fields (lower_snake).
# Only non-obvious mappings need to be listed; the rest are auto-converted.

_KEY_ALIASES = {
    "FOUNDATION_MODEL": "foundation_model",
    "SWIN_HEAD": "swin_head",
    "USE_QFORMER": "use_qformer",
    "USE_PTUNING": "use_ptuning",
    "USE_ORDINAL_CLASS": "use_ordinal_class",
    "USE_MULTILABEL": "use_multilabel",
    "USE_AUTOCAST": "use_autocast",
    "USE_DECODER": "use_decoder",
    "LABELS_KEY": "labels_key",
    "LABEL_SET": "label_set",
    "IMAGE_SIZE": "image_size",
    "DATA_PATH": "data_path",
    "TEST_DATA_PATH": "test_data_path",
    "SPLIT_FILE": "split_file",
    "BATCH_SIZE": "batch_size",
    "LEARNING_RATE": "learning_rate",
    "EPOCHS": "epochs",
    "NUM_FOLDS": "num_folds",
    "OVERSAMPLE": "oversample",
    "ALIGNMENT_LOSS_WEIGHT": "alignment_loss_weight",
    "FIDELITY_WEIGHT": "fidelity_weight",
    "FOUNDATION_MODEL_FROZEN_EPOCHS": "foundation_model_frozen_epochs",
    "SWIN_FROZEN_EPOCHS": "swin_frozen_epochs",
    "LEARNING_RATE_DECAY_STEP_SIZE": "learning_rate_decay_step_size",
    "LEARNING_RATE_DECAY_GAMMA": "learning_rate_decay_gamma",
    "CHECKPOINT_SAVE_DIR": "checkpoint_save_dir",
    "SWIN_PRETRAINED_PATH": "swin_pretrained_path",
    "PROJECT_NAME": "project_name",
    "GROUP_NAME": "group_name",
    "SWIN_PATCH_SIZE": "swin_patch_size",
    "SWIN_EMBED_DIM": "swin_embed_dim",
    "SWIN_DEPTHS": "swin_depths",
    "SWIN_NUM_HEADS": "swin_num_heads",
    "SWIN_WINDOW_SIZE": "swin_window_size",
    "DECODER_DEPTH": "decoder_depth",
    "DECODER_WINDOW_SIZE": "decoder_window_size",
    "SWIN_EXIT_STAGE": "swin_exit_stage",
    "SWIN_PHASE_EPOCHS": "swin_phase_epochs",
}


def _yaml_key_to_field(key: str) -> str:
    """Convert a YAML key to the dataclass field name."""
    if key in _KEY_ALIASES:
        return _KEY_ALIASES[key]
    return key.lower()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(
    config_path: str,
    overrides: Optional[dict] = None,
    resolve_paths: bool = True,
) -> ExperimentConfig:
    """Load an experiment config from a YAML file.

    Parameters
    ----------
    config_path : str
        Path to the YAML config file.
    overrides : dict, optional
        CLI overrides in {field_name: value} format (lower_snake keys).
    resolve_paths : bool
        If True, resolve data_path / test_data_path / split_file
        against BIOMED_DATA_DIR and validate they exist.

    Returns
    -------
    ExperimentConfig
    """
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    # Strip LABEL_DESCRIPTIONS from config — they live in labels.yaml now.
    # (Keeps backward compat with old config files that still have them.)
    raw.pop("LABEL_DESCRIPTIONS", None)

    # Build kwargs for the dataclass
    kwargs: dict = {}
    valid_fields = {f.name for f in ExperimentConfig.__dataclass_fields__.values()}

    for yaml_key, value in raw.items():
        field_name = _yaml_key_to_field(yaml_key)
        if field_name in valid_fields:
            kwargs[field_name] = value

    # Apply CLI overrides
    if overrides:
        for k, v in overrides.items():
            if k in valid_fields:
                kwargs[k] = v

    # Ensure numeric types
    if "learning_rate" in kwargs:
        kwargs["learning_rate"] = float(kwargs["learning_rate"])
    if "alignment_loss_weight" in kwargs:
        kwargs["alignment_loss_weight"] = float(kwargs["alignment_loss_weight"])
    if "fidelity_weight" in kwargs:
        kwargs["fidelity_weight"] = float(kwargs["fidelity_weight"])
    if "learning_rate_decay_gamma" in kwargs:
        kwargs["learning_rate_decay_gamma"] = float(kwargs["learning_rate_decay_gamma"])

    cfg = ExperimentConfig(**kwargs)

    # Load label descriptions
    all_labels = load_label_descriptions()
    if cfg.label_set:
        if cfg.label_set not in all_labels:
            available = ", ".join(sorted(all_labels.keys()))
            raise ValueError(
                f"Unknown label_set '{cfg.label_set}'. "
                f"Available sets in labels.yaml: {available}"
            )
        cfg.label_descriptions = all_labels[cfg.label_set]

    # Resolve paths against data root
    if resolve_paths and (cfg.data_path or cfg.split_file):
        data_root = get_data_root()
        cfg.resolved_data_path = _resolve_data_path(cfg.data_path, data_root, cfg.image_size)
        cfg.resolved_test_data_path = _resolve_data_path(
            cfg.test_data_path or cfg.data_path, data_root, cfg.image_size
        )
        if cfg.split_file:
            cfg.resolved_split_file = _resolve_data_path(cfg.split_file, data_root, cfg.image_size)

    # Resolve checkpoint dir
    cfg.resolved_checkpoint_dir = cfg.checkpoint_save_dir.format(
        project_name=cfg.project_name,
        group_name=cfg.group_name,
        image_size=cfg.image_size,
        suffix=cfg.image_size,
        fold_number="all",
    )

    return cfg


def _resolve_data_path(path_template: str, data_root: Path, image_size: int) -> str:
    """Resolve a path template.

    If the path is already absolute, use it as-is (backward compat).
    Otherwise treat it as relative to data_root.
    Substitutes {image_size} placeholder.
    """
    expanded = path_template.format(image_size=image_size)
    p = Path(expanded)
    if p.is_absolute():
        return str(p)
    return str(data_root / p)


def validate_data_files(cfg: ExperimentConfig) -> list[str]:
    """Check that all referenced data files exist.

    Returns a list of error messages (empty = all good).
    """
    errors = []
    for label, path in [
        ("data_path", cfg.resolved_data_path),
        ("test_data_path", cfg.resolved_test_data_path),
        ("split_file", cfg.resolved_split_file),
    ]:
        if path and not Path(path).exists():
            errors.append(f"  {label}: {path}")
    return errors

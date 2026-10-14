# AutoSwin

**Learned Downsampling Improves Fine-tuning of Foundation Models for Radiographic Classification of Small Disease Signs**

## Overview

Medical foundation models have achieved state-of-the-art performance across a wide range of diagnostic tasks, offering general-purpose representations that can be adapted to new problems with limited labelled data. However, computational constraints typically restrict their input to low resolutions (e.g., 224×224), and this downsampling is treated as a fixed pre-processing step. We show that such downsampling is insufficient for detecting fine-grained pathologies, where the diagnostic signal is concentrated in small structures such as the nodules associated with silicosis.

**AutoSwin** is a Hierarchical Shifted Window Transformer-based autoencoder that acts as a learned downsampling head for fixed-resolution foundation models, replacing standard interpolation without modifying the frozen encoder.

### Key Results

- **VinDr ChestX:** Overall diagnostic AUC improved from **0.850** to **0.869** (*p* = 0.002).
- **MBOD (Silicosis):** Silicosis detection AUC improved from **0.56** to **0.72**.
- The method significantly outperforms standard Lanczos interpolation for identifying small disease-signs across both datasets.

## Architecture

The proposed architecture consists of two stages:

### Stage I: AutoSwin Self-supervised Pre-training

A Swin Transformer encoder is trained to perform learned downsampling using a **Structure-Preserving Loss**:

```
L_struct = λ * L_fid + (1 - λ) * L_rec
```

- **Fidelity loss (`L_fid`)**: MSE between the encoder's low-resolution output and a Lanczos-interpolated target. Ensures the downsampled output maintains the pixel statistics and visual characteristics expected by the frozen foundation model vision encoder.
- **Reconstruction loss (`L_rec`)**: MSE between the decoder's high-resolution output and the original input. Encourages preservation of as much original signal as possible.

This stage requires only **unlabelled** X-ray images in their original resolution. Because diagnostic labels are not required, any set of X-rays preserved in their original DICOM format can be used, and the source and target datasets do not need overlapping labels.

### Stage II: Foundation Model Fine-tuning

The pre-trained AutoSwin encoder is attached to a frozen foundation model (e.g., BiomedCLIP). Three lightweight, trainable modules are introduced (see Figure 2 of the paper):

1. **AutoSwin Encoder Head**: The pre-trained learned downsampling module (high-resolution → low-resolution).
2. **Q-Former**: A query-transformer that compresses vision encoder outputs into a fixed set of query tokens for efficient adaptation.
3. **P-Tuning**: Prompt tuning on the text encoder to adapt class descriptions with minimal parameter overhead.

The total loss during fine-tuning at epoch *t* is:

```
L_total(t) = L_cls + γ(t) * L_struct
```

Where `γ(t)` is a scheduled decay coefficient that linearly reduces the influence of the structure-preserving objective over training epochs:

```
γ(t) = max(0, γ_init * (1 - t / T_sched))
```

This stabilises early training when the AutoSwin head is unfrozen, while allowing the model to focus fully on classification in later epochs. During inference, the decoder is discarded, meaning the AutoSwin head incurs negligible computational overhead compared to the backbone foundation model.

## Project Structure

```
autoswin/
├── __init__.py
├── models.py                  # Core architecture: FoundationModelWithSwin, SwinAutoencoderWithAdapter
├── swin_decoder.py            # Swin Transformer V2 decoder with PatchExpand modules
├── q_former.py                # Q-Former implementation
├── ptuned_text_encoder.py     # PEFT P-Tuning wrapper for the text encoder
├── txrv_wrapper.py            # TorchXRayVision ResNet wrapper (resolution baseline)
├── config.py                  # Unified configuration system (ExperimentConfig dataclass)
├── experiment.py              # Experiment tracking, checkpointing, and local logging
├── dataloader.py              # HDF5 dataset and PyTorch DataLoader utilities
├── metrics.py                 # Clinical metrics: sensitivity, specificity, AUC, etc.
├── multilabel_metrics.py      # Per-class multilabel metrics with AUC computation
├── focal_loss.py              # Focal loss implementation
├── utils.py                   # Visualisation and utility functions
└── labels.yaml                # Label descriptions for MBOD and VinDr datasets

scripts/
├── main.py                    # Main entry point: full k-fold Monte-Carlo cross validation
├── pretrain_swin.py           # Stage I: Pre-train the AutoSwin adapter
├── run_fold.py                # Run a single fold (for distributed / HPC execution)
├── run_fold_resolution.py     # Resolution baseline experiment with TorchXRayVision ResNet
└── visualise_downsamples.py   # Generate qualitative downsample comparisons

configs/
├── pretrain_swin.yaml                    # Standard pre-training config
├── pretrain_swin_progressive_896.yaml    # Progressive layer-wise pre-training
├── swin_train.yaml                       # Full fine-tuning config (VinDr)
├── mbod_baseline.yaml                    # Baseline without Swin (MBOD)
├── mbod_balanced_autoswin_896.yaml       # AutoSwin fine-tuning on MBOD
└── medclip_example.yaml                  # Example using MedCLIP as foundation model
```

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for Python environment and dependency management.

```bash
# Clone the repository
git clone <repo-url>
cd AutoSwin

# Create virtual environment and install dependencies
uv sync
```

Alternatively, with `pip`:

```bash
pip install -e .
```

### Dependencies

- Python >= 3.10
- PyTorch
- torchvision
- timm
- open-clip-torch
- h5py
- opencv-python
- scikit-learn / scikit-image
- matplotlib
- wandb (optional, for experiment tracking)
- torchxrayvision
- peft

## Usage

### Data Setup

Set the data directory environment variable:

```bash
export BIOMED_DATA_DIR=/path/to/your/data
```

This directory should contain:
- `.h5` dataset files (preprocessed images and labels)
- `.json` stratified split files

### Stage I: Pre-train AutoSwin

```bash
# Standard pre-training (fidelity-only, no decoder)
python scripts/pretrain_swin.py --config configs/pretrain_swin.yaml

# Progressive layer-wise pre-training (with decoder and fidelity/reconstruction balance)
python scripts/pretrain_swin.py --config configs/pretrain_swin_progressive_896.yaml

# Without WandB
python scripts/pretrain_swin.py --config configs/pretrain_swin.yaml --no-wandb
```

Pre-training checkpoints are saved to `experiments/<timestamp>-pretrain-swin/checkpoints/`.

### Stage II: Fine-tune Foundation Model

```bash
# Full k-fold Monte-Carlo Cross Validation
python scripts/main.py --config configs/swin_train.yaml

# Single fold (useful for HPC / distributed computing)
python scripts/run_fold.py --fold_number 0 --config configs/mbod_balanced_autoswin_896.yaml

# Without WandB
python scripts/main.py --config configs/swin_train.yaml --no-wandb
```

Each run creates an experiment directory under `experiments/` containing:
- `config.yaml` — snapshot of the configuration
- `git_info.json` — commit hash and dirty status
- `metrics.jsonl` — per-epoch metrics log
- `checkpoints/` — model checkpoints (best, latest, final)
- `figures/` — generated visualisations

### Resolution Baseline Experiment

To reproduce the resolution comparison (Section 4.1 of the paper) using the TorchXRayVision ResNet baseline:

```bash
python scripts/run_fold_resolution.py --fold_number 0 --image_size 224 --config configs/mbod_baseline.yaml
python scripts/run_fold_resolution.py --fold_number 0 --image_size 448 --config configs/mbod_baseline.yaml
python scripts/run_fold_resolution.py --fold_number 0 --image_size 896 --config configs/mbod_baseline.yaml
```

*Note: Ensure the baseline config has `SWIN_HEAD: False` so that the ResNet baseline is used.*

### Visualise Downsampling

Generate qualitative comparisons of Lanczos vs. AutoSwin downsampling:

```bash
python scripts/visualise_downsamples.py \
    --label_key silicosis \
    --model_path /path/to/checkpoints/ \
    --config_path configs/mbod_balanced_autoswin_896.yaml \
    --data_path $BIOMED_DATA_DIR/mbod_rand_balanced_896.h5
```

## Datasets

### Frontiers v1.1 baselines and smoke tests

BiomedCLIP now loads `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`.
Best checkpoints are selected from epoch 0 by validation **macro-averaged ROC AUC**
(classes without both outcomes are excluded), then reloaded for test evaluation.
`final-*` retains the last epoch; `best-foldN` is the selected checkpoint.
Zero-based fold N always uses seed `2026 + N`, even with a fixed stratified split.
Fixed split membership is preserved: these are repeated seeded runs, not new partitions.

The RAD-DINO baseline adapts the CLS + linear classifier in `../rad-dino/run.py`,
with its pinned Microsoft model revision, Lanczos preprocessing and end-to-end training.
It is self-contained: the sibling directory is not required at runtime.
Use `configs/raddino_mbod.yaml` with either CV entry point. It enables AMP,
batch size 1 and encoder gradient checkpointing for a 12 GB GPU.

```bash
uv run python scripts/smoke_test.py --model biomedclip
uv run python scripts/smoke_test.py --model raddino
uv run python scripts/main.py --config configs/raddino_mbod.yaml --no-wandb
```

Smoke tests download pretrained weights and run a synthetic optimizer step without data.

The experiments in the paper use two primary datasets:

- **MBOD**: South African Medical Bureau of Occupational Diseases dataset (private, 1,821 CXRs after quality filtering). Focused on silicosis and tuberculosis, with ILO (International Classification of Radiographs of Pneumoconioses) profusion scores.
- **VinDr ChestX**: Large-scale public dataset of 18,000 images with 28 labeled findings (available on [PhysioNet](https://physionet.org/) after ethics approval).

Label sets and class descriptions are defined in `autoswin/labels.yaml`.

## Configuration

All experiments are driven by YAML configuration files. Key parameters include:

| Parameter | Description |
|-----------|-------------|
| `FOUNDATION_MODEL` | Foundation model to use: `biomedclip` or `medclip` |
| `SWIN_HEAD` | Enable AutoSwin learned downsampling |
| `USE_QFORMER` | Enable Q-Former compression of vision features |
| `USE_PTUNING` | Enable P-Tuning on the text encoder |
| `USE_DECODER` | Enable decoder for reconstruction loss during pre-training |
| `IMAGE_SIZE` | Input resolution (e.g., 896 for high-resolution X-rays) |
| `FIDELITY_WEIGHT` | λ in structure-preserving loss (0.5 = balanced fidelity/reconstruction) |
| `SWIN_FROZEN_EPOCHS` | Number of epochs to keep AutoSwin frozen during fine-tuning |
| `NUM_FOLDS` | Number of folds for Monte-Carlo cross validation |

See the `configs/` directory for complete, working examples.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{bruton2024autoswin,
  title={Learned Downsampling Improves Fine-tuning of Foundation Models for Radiographic Classification of Small Disease Signs},
  author={Bruton, Joshua and Terespolsky, Sean and Sive, Warrick and Ehrlich, Rodney and Yassi, Annalee and Wang, Hairong and Klein, Richard},
  year={2024}
}
```

## License

This project is licensed under the GNU General Public License v3.0.

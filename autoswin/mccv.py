"""
Monte-Carlo Cross Validation trainer.

Refactored to use:
- ExperimentConfig (from config.py) instead of 15+ constructor args
- EvalResult dataclass instead of 11-element tuples
- Experiment for local results logging
- WandbLogger for optional WandB integration
"""

import glob
import os
import random
from contextlib import nullcontext
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from autoswin.config import ExperimentConfig
from autoswin.dataloader import get_dataloaders
from autoswin.experiment import EvalResult, Experiment, WandbLogger
from autoswin.metrics import (
    cosine_alignment_loss,
    get_accuracy,
    get_cm_for_class,
    get_f1_score,
    get_sensitivity,
    get_specificity,
    multilabel_specificity_at_sensitivity,
    multiclass_specificity_at_sensitivity,
    specificity_at_sensitivity,
)
from autoswin.multilabel_metrics import calculate_per_class_metrics
from autoswin.utils import (
    create_tsne_plot,
    ordinal_preds_to_class,
    plot_combined_conf_mat,
    plot_random_images,
    to_ordinal_labels,
)

# Reconstruction loss constants (used when fidelity_weight is active)
CRITERION = torch.nn.MSELoss()
RECON_WEIGHT = 0.2
RECON_WEIGHT_FACT = 0.05
RECON_WEIGHT_SCHED_EPOCH = 25


class MonteCarloCrossValidator:
    """Runs k-fold Monte-Carlo cross validation for BiomedCLIP experiments.

    Parameters
    ----------
    cfg : ExperimentConfig
        Full experiment configuration.
    model_init_func : callable
        Returns (model, preprocess) tuple.
    tokenizer : optional
        CLIP tokenizer. None for non-CLIP models (e.g. torchxrayvision).
    loss_fn : torch.nn.Module
        Loss function (BCEWithLogitsLoss or CrossEntropyLoss).
    device : torch.device
        Target device.
    experiment : Experiment
        Local experiment tracker.
    wandb_logger : WandbLogger
        Optional WandB wrapper.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        model_init_func,
        tokenizer,
        loss_fn,
        device: torch.device,
        experiment: Experiment,
        wandb_logger: WandbLogger,
    ):
        self.cfg = cfg
        self.model_init_func = model_init_func
        self.tokenizer = tokenizer
        self.loss_fn = loss_fn
        self.device = device
        self.exp = experiment
        self.wb = wandb_logger

        self.ampcontext = autocast if cfg.use_autocast else nullcontext
        self.scaler = GradScaler()

        if cfg.use_ordinal_class:
            self.num_classes = len(cfg.label_descriptions) + 1
        else:
            self.num_classes = len(cfg.label_descriptions)

        # Resolved paths from config
        self.hdf5_path = cfg.resolved_data_path
        self.test_hdf5_path = cfg.resolved_test_data_path or self.hdf5_path
        self.split_file = cfg.resolved_split_file

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _get_dataloaders(self, train_split=None, iteration=None):
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

        if self.hdf5_path == self.test_hdf5_path:
            if self.split_file:
                print(f"Loading from split files: {self.split_file}")
                augmentations = T.Compose([
                    T.RandomRotation(degrees=10, expand=False, fill=0),
                    T.RandomAffine(degrees=0, translate=(0.05, 0.05), fill=0),
                ])

                train_loader, _, _ = get_dataloaders(
                    hdf5_path=self.hdf5_path,
                    preprocess=self.preprocess,
                    train_split=0,
                    batch_size=self.cfg.batch_size,
                    labels_key=self.cfg.labels_key,
                    split_file=self.split_file,
                    augmentations=augmentations,
                    oversample=self.cfg.oversample,
                )

                _, val_loader, test_loader = get_dataloaders(
                    hdf5_path=self.hdf5_path,
                    preprocess=self.preprocess,
                    train_split=0,
                    batch_size=1,
                    labels_key=self.cfg.labels_key,
                    split_file=self.split_file,
                    augmentations=None,
                    oversample=False,
                )
            else:
                if train_split is None:
                    train_split = random.randint(40, 60) / 100

                train_loader, val_loader, test_loader = get_dataloaders(
                    hdf5_path=self.hdf5_path,
                    preprocess=self.preprocess,
                    train_split=train_split,
                    batch_size=self.cfg.batch_size,
                    labels_key=self.cfg.labels_key,
                    split_file=f"data_splits/{timestamp}_{iteration}.json",
                    oversample=self.cfg.oversample,
                )
        else:
            train_loader, _, _ = get_dataloaders(
                hdf5_path=self.hdf5_path,
                preprocess=self.preprocess,
                train_split=1,
                batch_size=self.cfg.batch_size,
                labels_key=self.cfg.labels_key,
                split_file=f"data_splits/training-{timestamp}_{iteration}.json",
                oversample=self.cfg.oversample,
            )

            _, val_loader, test_loader = get_dataloaders(
                hdf5_path=self.test_hdf5_path,
                preprocess=self.preprocess,
                train_split=0,
                batch_size=1,
                labels_key=self.cfg.labels_key,
                split_file=f"data_splits/testing-{timestamp}_{iteration}.json",
                oversample=False,
            )

        return train_loader, val_loader, test_loader

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, dataloader, epoch, fold, prefix="", log=True) -> EvalResult:
        self.model.eval()

        global_confusion = None
        all_gt = []
        all_probs = []
        all_image_features = []
        all_text_features = []
        all_labels = []

        with torch.no_grad():
            if self.tokenizer:
                with self.ampcontext():
                    text_features = self.model.encode_text(
                        self.tokenizer(list(self.cfg.label_descriptions), context_length=256).to(self.device),
                        normalize=True,
                    )
            else:
                text_features = None

            for batch_images, batch_labels in tqdm(dataloader):
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)

                with self.ampcontext():
                    logits, image_features = self.dpmodel(batch_images, text_features, return_image_features=True)

                if self.cfg.use_multilabel:
                    probs = torch.sigmoid(logits)
                else:
                    probs = torch.softmax(logits, dim=1)

                assert not torch.isnan(probs).any(), f"NaN in probs, logits: {logits}"
                assert not torch.isinf(probs).any(), f"Inf in probs, logits: {logits}"

                all_gt.append(batch_labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

                # Collect features for t-SNE (capped at 1000)
                if len(all_image_features) < 1000:
                    all_image_features.append(image_features.cpu().numpy())
                    cpu_labels = batch_labels.cpu().numpy()
                    if self.cfg.use_multilabel and len(batch_labels.shape) > 1:
                        all_labels.extend(np.argmax(cpu_labels, axis=1))
                    else:
                        all_labels.extend(cpu_labels)

            if text_features is not None:
                all_text_features = text_features.cpu().numpy()

        all_gt = np.concatenate(all_gt, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)

        # Specificity @ 0.9 sensitivity
        if self.cfg.use_multilabel:
            spec_at_09, thresh = multilabel_specificity_at_sensitivity(all_gt, all_probs, min_sens=0.9)
        elif self.num_classes == 2:
            spec_at_09, thresh = specificity_at_sensitivity(all_gt, all_probs[:, 1], min_sens=0.9)
        else:
            spec_at_09, thresh = multiclass_specificity_at_sensitivity(all_gt, all_probs, min_sens=0.9)

        if log:
            self.wb.log({f"{prefix}spec_at_09_sens": spec_at_09}, step=epoch)

        # Predictions
        if self.cfg.use_ordinal_class:
            predictions = ordinal_preds_to_class(all_probs)
        elif self.cfg.use_multilabel:
            predictions = (all_probs > thresh)
        else:
            predictions = np.argmax(all_probs, axis=1)

        # Confusion matrix
        if self.cfg.use_multilabel:
            global_confusion = multilabel_confusion_matrix(all_gt, predictions)
        else:
            global_confusion = confusion_matrix(all_gt, predictions, labels=list(range(self.num_classes)))

        # t-SNE plot
        if epoch % 1 == 0 and text_features is not None:
            tsne_fig = create_tsne_plot(all_image_features, all_text_features, all_labels)
            if log:
                self.wb.log_image(f"{prefix}tsne_plot", tsne_fig, step=epoch)
                self.exp.save_figure(tsne_fig, f"{prefix}tsne", epoch=epoch)
            plt.close(tsne_fig)

        accuracy = get_accuracy(global_confusion)
        sensitivity = get_sensitivity(global_confusion)
        specificity = get_specificity(global_confusion)
        f1 = get_f1_score(global_confusion)

        # Swin downsample visualisation
        if self.model.swin_with_adapter is not None:
            fig = plot_random_images(dataloader, self.model, self.device)
            if log:
                self.wb.log_image(f"{prefix}swin_downsamples", fig, step=epoch)
                self.exp.save_figure(fig, f"{prefix}swin_downsamples", epoch=epoch)
            plt.close(fig)

        # Multilabel per-class metrics
        ml_metrics_dict = None
        average_auc = None
        if self.cfg.use_multilabel:
            ml_metrics_dict, average_auc = calculate_per_class_metrics(
                global_confusion,
                prefix=prefix,
                y_true=all_gt,
                y_prob=all_probs,
                class_names=self.cfg.label_descriptions,
            )
            if log:
                self.wb.log(ml_metrics_dict, step=epoch)

        # Confusion matrix figure
        conf_fig = plot_combined_conf_mat(global_confusion)

        # Binary metrics for multiclass (class 0 vs rest)
        bin_accuracy = bin_sensitivity = bin_specificity = bin_f1 = None
        if global_confusion.shape != (2, 2) and not self.cfg.use_multilabel:
            TN, FP, FN, TP = get_cm_for_class(global_confusion, 0)
            binary_cm = np.array([[TP, FN], [FP, TN]])
            bin_accuracy = get_accuracy(binary_cm)
            bin_sensitivity = get_sensitivity(binary_cm)
            bin_specificity = get_specificity(binary_cm)
            bin_f1 = get_f1_score(binary_cm)

        result = EvalResult(
            accuracy=accuracy,
            sensitivity=sensitivity,
            specificity=specificity,
            f1=f1,
            bin_accuracy=bin_accuracy,
            bin_sensitivity=bin_sensitivity,
            bin_specificity=bin_specificity,
            bin_f1=bin_f1,
            average_auc=average_auc,
            ml_metrics_dict=ml_metrics_dict,
            confusion_fig=conf_fig,
        )

        return result

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_eval_metrics(self, prefix: str, epoch: int, fold: int, result: EvalResult, loss: float = None):
        """Log evaluation metrics to both WandB and local experiment."""
        metrics = {
            f"{prefix}acc": result.accuracy,
            f"{prefix}sens": result.sensitivity,
            f"{prefix}spec": result.specificity,
            f"{prefix}f1": result.f1,
        }
        if result.average_auc is not None:
            metrics[f"{prefix}auc"] = result.average_auc
        if result.bin_accuracy is not None:
            metrics[f"{prefix}bin_acc"] = result.bin_accuracy
            metrics[f"{prefix}bin_sens"] = result.bin_sensitivity
            metrics[f"{prefix}bin_spec"] = result.bin_specificity
            metrics[f"{prefix}bin_f1"] = result.bin_f1
        if loss is not None:
            metrics[f"epoch_{prefix}loss"] = loss
            metrics["fold"] = fold

        self.wb.log(metrics, step=epoch)
        if result.confusion_fig is not None:
            self.wb.log_image(f"{prefix}confusion_matrix", result.confusion_fig, step=epoch)

        # Local logging
        self.exp.log_eval(prefix.rstrip("_"), epoch=epoch, result=result, fold=fold)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, train_split=None, iteration=None, start_epoch=0) -> EvalResult:
        peak_validation_spec = 0
        start_checkpoint_saves = 50
        patience = 1000
        min_delta = 0.001
        epochs_without_improvement = 0

        train_loader, val_loader, test_loader = self._get_dataloaders(train_split, iteration)

        # Initial swin visualisation
        if self.model.swin_with_adapter is not None:
            print("Generating initial downsamples...")
            fig = plot_random_images(val_loader, self.model, self.device)
            self.wb.log_image("initial_swin_downsamples", fig, step=0)
            self.exp.save_figure(fig, "initial_swin_downsamples")
            plt.close(fig)

        self.model.freeze_weights()

        # Freeze swin weights if configured
        if self.cfg.swin_frozen_epochs != 0:
            self.model.freeze_swin_weights(False)

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)
        self.model.print_parameter_count()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.cfg.learning_rate_decay_step_size,
            gamma=self.cfg.learning_rate_decay_gamma,
        )

        epoch = start_epoch  # ensure epoch is defined even if loop doesn't run
        for epoch in tqdm(range(start_epoch, self.cfg.epochs), desc="Training...", unit="epoch"):
            recon_weight = round(RECON_WEIGHT * (1 - ((epoch // RECON_WEIGHT_SCHED_EPOCH) * RECON_WEIGHT_FACT)), 2)

            if self.cfg.swin_frozen_epochs != 0 and self.cfg.swin_frozen_epochs <= epoch:
                self.model.freeze_swin_weights(True)

            self.model.train()
            epoch_loss = 0
            epoch_recon_loss = 0
            total = 0

            for batch_images, batch_labels in tqdm(train_loader):
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)

                with self.ampcontext():
                    if self.tokenizer:
                        text_features = self.model.encode_text(
                            self.tokenizer(list(self.cfg.label_descriptions), context_length=256).to(self.device),
                            normalize=True,
                        )
                    else:
                        text_features = None

                    logits, image_features = self.dpmodel(batch_images, text_features, return_image_features=True)

                    if self.cfg.use_ordinal_class:
                        loss_labels = to_ordinal_labels(batch_labels, num_classes=self.num_classes)
                    else:
                        loss_labels = batch_labels

                    if self.cfg.use_multilabel:
                        loss_labels = loss_labels.float()

                    loss = self.loss_fn(logits, loss_labels)

                    if self.cfg.alignment_loss_weight > 0:
                        alignment_loss = cosine_alignment_loss(
                            image_features, text_features, batch_labels, self.cfg.use_multilabel
                        )
                        loss = ((1 - self.cfg.alignment_loss_weight) * loss) + (self.cfg.alignment_loss_weight * alignment_loss)

                    # Reconstruction loss (for Swin fidelity training)
                    recon_loss = None
                    if self.cfg.fidelity_weight != -1 and recon_weight > 0:
                        if self.cfg.fidelity_weight == 1:
                            ds = self.model.swin_with_adapter(batch_images)
                        else:
                            ds, recon = self.model.swin_with_adapter(batch_images, decoder_pass=True)

                        # Fidelity target at actual output resolution (supports progressive stages)
                        target_size = ds.shape[-2:]
                        target_images = F.interpolate(
                            batch_images, size=target_size, mode='bilinear', align_corners=False
                        )
                        if target_images.shape[1] == 1:
                            target_images = target_images.repeat(1, 3, 1, 1)

                        if self.cfg.fidelity_weight == 1:
                            recon_loss = CRITERION(ds, target_images)
                        else:
                            fidelity_loss = CRITERION(ds, target_images)
                            rec_loss = CRITERION(recon, batch_images)
                            recon_loss = (self.cfg.fidelity_weight * fidelity_loss) + ((1 - self.cfg.fidelity_weight) * rec_loss)

                    epoch_loss += loss.item()

                    if recon_loss is not None and recon_weight > 0:
                        loss = loss + (recon_weight * recon_loss)
                        epoch_recon_loss += recon_loss.item()

                    total += batch_labels.size(0)

                if self.cfg.use_autocast:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
                    self.optimizer.step()

            self.scheduler.step()

            # --- Train evaluation ---
            train_result = self.evaluate(train_loader, epoch, iteration, prefix="train_")
            train_loss = epoch_loss / total if total > 0 else 0
            self._log_eval_metrics("train_", epoch, iteration, train_result, loss=train_loss)

            # Log recon loss separately
            if epoch_recon_loss > 0:
                self.wb.log({"epoch_train_recon_loss": epoch_recon_loss / total}, step=epoch)

            plt.close(train_result.confusion_fig)

            # --- Validation evaluation ---
            val_result = self.evaluate(val_loader, epoch, iteration)
            self._log_eval_metrics("val_", epoch, iteration, val_result)
            plt.close(val_result.confusion_fig)

            # --- Checkpointing ---
            if epoch > start_checkpoint_saves:
                if val_result.specificity > peak_validation_spec + min_delta:
                    peak_validation_spec = val_result.specificity
                    epochs_without_improvement = 0
                    self.exp.save_checkpoint(
                        self.model.state_dict(),
                        tag=f"best-epoch{epoch}-spec{val_result.specificity:.4f}",
                    )
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                        break
            elif val_result.specificity > peak_validation_spec:
                peak_validation_spec = val_result.specificity

            # Save latest checkpoint (overwrite previous)
            self.exp.remove_checkpoints("latest")
            self.exp.save_checkpoint(self.model.state_dict(), tag=f"latest-{epoch}")

        # --- Test evaluation ---
        test_result = self.evaluate(test_loader, epoch, iteration, prefix="test_")

        # Save final model
        tag = f"final-fold{iteration}" if iteration is not None else "final"
        self.exp.save_checkpoint(self.model.state_dict(), tag=tag)
        print("Training complete. Model saved.")

        return test_result

    # ------------------------------------------------------------------
    # K-fold runner
    # ------------------------------------------------------------------

    def run_k_iterations(self, k: int, suffix: str = ""):
        accuracies = []
        sensitivities = []
        specificities = []

        import wandb as _wandb
        group_name = f"local-{self.cfg.checkpoint_save_dir}-{suffix}" + _wandb.util.generate_id() if self.wb.enabled else f"local-{suffix}"

        for i in range(k):
            self.dpmodel, self.preprocess = self.model_init_func()
            if torch.cuda.device_count() > 1:
                self.dpmodel = torch.nn.DataParallel(self.dpmodel)
            self.dpmodel.to(self.device)
            self.dpmodel.train()

            self.model = self.dpmodel.module if isinstance(self.dpmodel, torch.nn.DataParallel) else self.dpmodel

            run_name = f"fold-{i}"
            self.wb.init(
                project=self.cfg.project_name,
                group=group_name,
                name=run_name,
                job_type=run_name,
                config={
                    "loss_function": self.loss_fn.__class__.__name__,
                    "optimizer": "AdamW",
                    "learning_rate": self.cfg.learning_rate,
                    "learning_rate_decay_step_size": self.cfg.learning_rate_decay_step_size,
                    "learning_rate_decay_gamma": self.cfg.learning_rate_decay_gamma,
                    "dataset": self.hdf5_path,
                    "labels_key": self.cfg.labels_key,
                    "batch_size": self.cfg.batch_size,
                    "epochs": self.cfg.epochs,
                    "fold": i,
                    "ordinal_class": self.cfg.use_ordinal_class,
                    "foundation_model": self.cfg.foundation_model,
                    "save_directory": self.exp.dir.as_posix(),
                },
            )

            test_result = self.train(iteration=i)
            accuracies.append(test_result.accuracy)
            sensitivities.append(test_result.sensitivity)
            specificities.append(test_result.specificity)

            # Log final test metrics
            self._log_eval_metrics("test_", epoch=0, fold=i, result=test_result)
            self.exp.log_fold_summary(fold=i, test_result=test_result)

            plt.close(test_result.confusion_fig)
            self.wb.finish()

        avg_accuracy = np.mean(accuracies)
        avg_sensitivity = np.mean(sensitivities)
        avg_specificity = np.mean(specificities)

        print(f"\nResults over {k} iterations:")
        print(f"Average Test Accuracy: {avg_accuracy:.2f}")
        print(f"Average Sensitivity (True Positive Rate): {avg_sensitivity:.2f}")
        print(f"Average Specificity (True Negative Rate): {avg_specificity:.2f}")

        print("Per run stats (accuracy, sensitivity, specificity):")
        print(accuracies)
        print(sensitivities)
        print(specificities)

        # Save summary
        self.exp.log_metrics(
            epoch=-1,
            type="cross_validation_summary",
            avg_accuracy=avg_accuracy,
            avg_sensitivity=avg_sensitivity,
            avg_specificity=avg_specificity,
            per_fold_accuracy=accuracies,
            per_fold_sensitivity=sensitivities,
            per_fold_specificity=specificities,
        )

        return avg_accuracy, avg_sensitivity, avg_specificity

    # ------------------------------------------------------------------
    # Single fold runner (for distributed / HPC)
    # ------------------------------------------------------------------

    def run_fold(self, fold_number: int, group_name: str):
        self.dpmodel, self.preprocess = self.model_init_func()
        if torch.cuda.device_count() > 1:
            self.dpmodel = torch.nn.DataParallel(self.dpmodel)
        self.dpmodel.to(self.device)
        self.dpmodel.train()

        self.model = self.dpmodel.module if isinstance(self.dpmodel, torch.nn.DataParallel) else self.dpmodel

        # Resume from latest checkpoint if available
        latest = None
        for f in glob.glob(str(self.exp.checkpoints_dir / "latest-*.pth")):
            latest = f

        latest_epoch = -1
        if latest:
            pretrained_dict = torch.load(latest)
            self.model.load_state_dict(pretrained_dict)
            # Extract epoch number from filename
            basename = os.path.basename(latest)
            latest_epoch = int(basename.replace("latest-", "").replace(".pth", ""))
            print(f"Found {latest}... starting from epoch {latest_epoch + 1}...")

        run_name = f"fold-{fold_number}"
        self.wb.init(
            project=self.cfg.project_name,
            group=group_name,
            name=run_name,
            job_type=run_name,
            config={
                "loss_function": self.loss_fn.__class__.__name__,
                "optimizer": "AdamW",
                "learning_rate": self.cfg.learning_rate,
                "learning_rate_decay_step_size": self.cfg.learning_rate_decay_step_size,
                "learning_rate_decay_gamma": self.cfg.learning_rate_decay_gamma,
                "dataset": self.hdf5_path,
                "labels_key": self.cfg.labels_key,
                "batch_size": self.cfg.batch_size,
                "epochs": self.cfg.epochs,
                "fold": fold_number,
                "ordinal_class": self.cfg.use_ordinal_class,
                "foundation_model": self.cfg.foundation_model,
                "save_directory": self.exp.dir.as_posix(),
            },
        )

        test_result = self.train(iteration=fold_number, start_epoch=latest_epoch + 1)

        self._log_eval_metrics("test_", epoch=0, fold=fold_number, result=test_result)
        self.exp.log_fold_summary(fold=fold_number, test_result=test_result)

        plt.close(test_result.confusion_fig)
        self.wb.finish()

    # ------------------------------------------------------------------
    # Checkpoint evaluation (no training)
    # ------------------------------------------------------------------

    def load_and_evaluate_checkpoint(self, checkpoint_path: str) -> EvalResult:
        self.dpmodel, self.preprocess = self.model_init_func()
        if torch.cuda.device_count() > 1:
            self.dpmodel = torch.nn.DataParallel(self.dpmodel)
        self.dpmodel.to(self.device)
        self.dpmodel.train()

        self.model = self.dpmodel.module if isinstance(self.dpmodel, torch.nn.DataParallel) else self.dpmodel

        pretrained_dict = torch.load(checkpoint_path)
        self.model.load_state_dict(pretrained_dict)

        _, val_loader, test_loader = self._get_dataloaders(None, None)

        return self.evaluate(val_loader, 0, 0, prefix="eval_", log=False)

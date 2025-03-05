"""
PyTorch Lightning module for model training, validation, and testing.

This module provides:
- Base training logic for various computer vision tasks
- Metrics tracking and logging
- Visualization capabilities
- Customizable loss functions

Customize this file to fit your specific task and metrics needs.
"""

from pdb import set_trace
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics import MetricCollection
from torchmetrics.classification import Precision, Recall, F1Score
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError


class BaseTask(pl.LightningModule):
    """Base Lightning Module for training models."""

    def __init__(
        self,
        model,
        loss="mse",
        learning_rate=0.001,
        weight_decay=0.01,
        optimizer="adamw",
        scheduler="cosine",
        scheduler_params=None,
        **kwargs,
    ):
        """Initialize the Lightning Module.

        Args:
            model: PyTorch model to train
            loss: Loss function to use
            learning_rate: Initial learning rate
            weight_decay: Weight decay factor
            optimizer: Optimizer to use (adam, adamw, sgd, rmsprop)
            scheduler: Learning rate scheduler (cosine, plateau, step, none)
            scheduler_params: Additional parameters for the scheduler
            **kwargs: Additional arguments
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.configure_loss(loss)
        self.configure_metrics()

    def configure_loss(self, loss_name):
        """Configure loss function.

        Args:
            loss_name: Name of the loss function
        """
        if loss_name == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_name == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif loss_name == "ce":
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_name == "huber":
            self.loss_fn = nn.HuberLoss(delta=0.7)
        elif loss_name == "dice":
            # Add more specialized losses as needed
            self.loss_fn = nn.MSELoss()  # Placeholder
        else:
            raise ValueError(f"Unsupported loss: {loss_name}")

    def forward(self, x):
        """Forward pass."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step.

        Args:
            batch: Current batch
            batch_idx: Index of current batch

        Returns:
            Loss value
        """
        x, y = batch["image"], batch["mask"]
        y_hat = self(x)

        if y.dim() == 4 and y.size(1) == 1:
            y = y.squeeze(1)

        # Convert target to Long type for CE loss
        y = y.long()

        # y_hat should remain [B, C, H, W] for cross entropy loss
        loss = self.loss_fn(y_hat, y)

        # For metrics, ensure predictions and targets have compatible shapes
        # For multi-class segmentation, convert predictions to class indices
        if y_hat.size(1) > 1 and y.dim() < 4:  # Multi-class case
            y_hat_for_metrics = y_hat.argmax(dim=1)  # [B, H, W]
        else:
            y_hat_for_metrics = y_hat

        # Update and log metrics
        self.train_metrics(y_hat_for_metrics, y)
        self.log("train_loss", loss)
        self.log_dict(self.train_metrics)

        # Visualize training examples occasionally
        if batch_idx % 100 == 0:
            self.visualize_batch(x, y, y_hat, "train", batch_idx)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step.

        Args:
            batch: Current batch
            batch_idx: Index of current batch
        """
        x, y = batch["image"], batch["mask"]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        if y_hat.size(1) > 1 and y.dim() < 4:
            y_hat_for_metrics = y_hat.argmax(dim=1)
        else:
            y_hat_for_metrics = y_hat

        # Update and log metrics
        self.val_metrics(y_hat_for_metrics, y)
        self.log("val_loss", loss)
        self.log_dict(self.val_metrics)

        # Visualize validation examples occasionally
        if batch_idx == 0:
            self.visualize_batch(x, y, y_hat, "val", self.current_epoch)

    def test_step(self, batch, batch_idx):
        """Test step.

        Args:
            batch: Current batch
            batch_idx: Index of current batch
        """
        x, y = batch["image"], batch["mask"]
        y_hat = self(x)
        if y.dim() == 4 and y.size(1) == 1:
            y = y.squeeze(1)
        y = y.long()
        if y_hat.size(1) > 1 and y.dim() < 4:
            y_hat_for_metrics = y_hat.argmax(dim=1)  # [B, H, W]
        else:
            y_hat_for_metrics = y_hat

        # Update and log metrics
        self.test_metrics(y_hat_for_metrics, y)
        self.log_dict(self.test_metrics)

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Select optimizer
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                momentum=0.9,
            )
        elif self.hparams.optimizer == "rmsprop":
            optimizer = torch.optim.RMSprop(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer}")

        # Configure scheduler
        if self.hparams.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.get("t_max", 10), eta_min=1e-6
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss",
            }

        elif self.hparams.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=self.hparams.get("patience", 10),
                factor=self.hparams.get("factor", 0.1),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss",
            }

        elif self.hparams.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.hparams.get("step_size", 30),
                gamma=self.hparams.get("gamma", 0.1),
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        elif self.hparams.scheduler == "none" or self.hparams.scheduler is None:
            return {"optimizer": optimizer}

        else:
            raise ValueError(f"Unsupported scheduler: {self.hparams.scheduler}")


class SegmentationTask(BaseTask):
    """Task for semantic segmentation.

    This class serves as a wrapper around TorchGeo's SemanticSegmentationTask.
    """

    def __init__(self, modelc, *args, **kwargs):
        """Initialize segmentation task.

        Args:
            model: Model to use (RandomModel or None for TorchGeo models)
            *args, **kwargs: Additional arguments
        """
        # Import TorchGeo's semantic segmentation task
        from torchgeo.trainers import SemanticSegmentationTask

        # For RandomModel, we use our own implementation
        if modelc.__class__.__name__ == "RandomModel":
            super().__init__(modelc, *args, **kwargs)
            self.is_random_model = True
        else:
            # Use TorchGeo's implementation for standard models
            # Extract parameters for TorchGeo
            model_name = kwargs.pop("model_name", None)
            backbone_name = kwargs.pop("backbone_name", None)
            weight_init = kwargs.pop("weight_init", None)
            weights = True if weight_init == "pretrained" else False
            in_channels = kwargs.pop("in_channels", None)
            out_channels = kwargs.pop("out_channels", None)
            loss_name = kwargs.pop("loss", None)
            lr = kwargs.pop("learning_rate", None)

            # First create TorchGeo task - call parent's __init__ with this as the model
            torchgeo_task = SemanticSegmentationTask(
                model=model_name,
                backbone=backbone_name,
                weights=weights,
                in_channels=in_channels,
                num_classes=out_channels,
                loss=loss_name,
                lr=lr,
            )

            # Call super().__init__() with the TorchGeo task's model
            super().__init__(torchgeo_task.model, *args, **kwargs)

            # Now assign the task after parent initialization is complete
            self.torchgeo_task = torchgeo_task
            self.is_random_model = False

            # For access to hyperparameters
            param_dict = {}
            for key, value in kwargs.items():
                param_dict[key] = value

            # Add model parameters for completeness
            param_dict["model_name"] = model_name
            param_dict["backbone_name"] = backbone_name
            param_dict["weight_init"] = weight_init
            param_dict["in_channels"] = in_channels
            param_dict["out_channels"] = out_channels
            param_dict["loss"] = loss_name
            param_dict["learning_rate"] = lr

            # Save hyperparameters properly
            self.save_hyperparameters(param_dict)

    def configure_metrics(self):
        """Configure metrics for segmentation evaluation.

        Use classification metrics appropriate for segmentation tasks.
        """
        # Determine task type and number of classes
        num_classes = self.hparams.get("out_channels", 2)
        task_type = "binary" if num_classes <= 2 else "multiclass"

        # Configure metrics for each stage
        self.train_metrics = MetricCollection(
            {
                "train_precision": Precision(task=task_type, num_classes=num_classes),
                "train_recall": Recall(task=task_type, num_classes=num_classes),
                "train_f1": F1Score(task=task_type, num_classes=num_classes),
            }
        )

        self.val_metrics = MetricCollection(
            {
                "val_precision": Precision(task=task_type, num_classes=num_classes),
                "val_recall": Recall(task=task_type, num_classes=num_classes),
                "val_f1": F1Score(task=task_type, num_classes=num_classes),
            }
        )

        self.test_metrics = MetricCollection(
            {
                "test_precision": Precision(task=task_type, num_classes=num_classes),
                "test_recall": Recall(task=task_type, num_classes=num_classes),
                "test_f1": F1Score(task=task_type, num_classes=num_classes),
            }
        )

    def forward(self, x):
        """Forward pass."""
        if self.is_random_model:
            return super().forward(x)
        else:
            return self.torchgeo_task(x)

    def training_step(self, batch, batch_idx):
        """Training step."""
        if self.is_random_model:
            return super().training_step(batch, batch_idx)
        else:
            return self.torchgeo_task.training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        if self.is_random_model:
            return super().validation_step(batch, batch_idx)
        else:
            return self.torchgeo_task.validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        """Test step."""
        if self.is_random_model:
            return super().test_step(batch, batch_idx)
        else:
            return self.torchgeo_task.test_step(batch, batch_idx)

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        if self.is_random_model:
            return super().configure_optimizers()
        else:
            return self.torchgeo_task.configure_optimizers()

    def visualize_batch(self, x, y, y_hat, stage, idx):
        """Visualization for segmentation tasks."""

        # x.shape == 4, 1, 9, 512, 512
        # y.shape == 4, 512, 512
        # y_hat.shape == 4, 2, 512, 512

        # Squeeze x & Get the first 3 channels
        x = x.squeeze(1)[:, :3]  # 4, 3, 512, 512

        # Do argmax on y_hat
        y_hat = y_hat.argmax(dim=1)  # 4, 512, 512

        # what are the expected shapes?
        # x.shape == 4, 3, 512, 512
        # y.shape == 4, 512, 512
        # y_hat.shape == 4, 512, 512

        # Min-max normalize the images one by one
        for i in range(x.size(0)):
            x[i] = (x[i] - x[i].min()) / (x[i].max() - x[i].min())

        # Convert x, y, and y_hat to numpy arrays
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        y_hat = y_hat.cpu().numpy()

        n_samples = min(4, x.shape[0])

        fig, axs = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
        if n_samples == 1:
            axs = axs.reshape(1, -1)

        for i in range(n_samples):

            # Input image (RGB)
            axs[i, 0].imshow(np.transpose(x[i], (1, 2, 0)))
            axs[i, 0].set_title("Input Image")
            axs[i, 0].axis("off")

            # Ground truth mask
            axs[i, 1].imshow(y[i], cmap="tab20")
            axs[i, 1].set_title("Ground Truth")
            axs[i, 1].axis("off")

            # Predicted mask
            axs[i, 2].imshow(y_hat[i], cmap="tab20")
            axs[i, 2].set_title("Prediction")
            axs[i, 2].axis("off")

        plt.tight_layout()

        # Log the figure if logger is available
        if hasattr(self, "logger") and self.logger is not None:
            self.logger.experiment.add_figure(
                f"{stage}_segmentation_{idx}", fig, global_step=self.global_step
            )

        plt.close(fig)


class RegressionTask(BaseTask):
    """Task for regression problems."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional regression metrics can be added here

    def configure_metrics(self):
        """Configure metrics for regression tasks.

        Use appropriate regression metrics (MSE, MAE).
        """
        self.train_metrics = MetricCollection(
            {"train_mse": MeanSquaredError(), "train_mae": MeanAbsoluteError()}
        )

        self.val_metrics = MetricCollection(
            {"val_mse": MeanSquaredError(), "val_mae": MeanAbsoluteError()}
        )

        self.test_metrics = MetricCollection(
            {"test_mse": MeanSquaredError(), "test_mae": MeanAbsoluteError()}
        )

    def visualize_batch(self, x, y, y_hat, stage, idx):
        """Override to customize visualization for regression tasks."""
        if not hasattr(self, "logger") or self.logger is None:
            return

        n_samples = min(4, x.size(0))

        fig, axs = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
        if n_samples == 1:
            axs = axs.reshape(1, -1)

        for i in range(n_samples):
            # Input image
            img = x[i, :3].permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)
            axs[i, 0].imshow(img)
            axs[i, 0].set_title("Input")
            axs[i, 0].axis("off")

            # Target continuous values
            target = y[i, 0].cpu().numpy()
            im = axs[i, 1].imshow(target, cmap="viridis")
            axs[i, 1].set_title("Target Values")
            axs[i, 1].axis("off")
            fig.colorbar(im, ax=axs[i, 1])

            # Predicted values
            pred = y_hat[i, 0].detach().cpu().numpy()
            im = axs[i, 2].imshow(pred, cmap="viridis")
            axs[i, 2].set_title("Predicted Values")
            axs[i, 2].axis("off")
            fig.colorbar(im, ax=axs[i, 2])

        plt.tight_layout()
        self.logger.experiment.add_figure(
            f"{stage}_regression_{idx}", fig, global_step=self.global_step
        )
        plt.close(fig)


def get_task(task_type, model, **kwargs):
    """Factory function to get task by type.

    Args:
        task_type: Type of task (segmentation, regression, classification)
        model: Model to use
        **kwargs: Additional arguments for the task

    Returns:
        LightningModule instance
    """
    # Get kwargs specific to the task type
    model_name = kwargs.pop("model_name", "unet")

    if task_type.lower() == "segmentation":
        return SegmentationTask(model, model_name=model_name, **kwargs)
    elif task_type.lower() == "regression":
        return RegressionTask(model, **kwargs)
    elif task_type.lower() == "base":
        return BaseTask(model, **kwargs)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

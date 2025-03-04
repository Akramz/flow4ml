"""
PyTorch Lightning module for model training, validation, and testing.

This module provides:
- Base training logic for various computer vision tasks
- Metrics tracking and logging
- Visualization capabilities
- Customizable loss functions

Customize this file to fit your specific task and metrics needs.
"""

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
        **kwargs
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
    
    def configure_metrics(self):
        """Configure metrics for evaluation."""
        # Customize or extend these metrics for your specific task
        self.train_metrics = MetricCollection({
            'train_mse': MeanSquaredError(),
            'train_mae': MeanAbsoluteError(),
        })
        
        self.val_metrics = MetricCollection({
            'val_mse': MeanSquaredError(),
            'val_mae': MeanAbsoluteError(),
        })
        
        self.test_metrics = MetricCollection({
            'test_mse': MeanSquaredError(),
            'test_mae': MeanAbsoluteError(),
        })
        
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
        # Customize this method for your data format
        x, y = batch["image"], batch["target"]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        # Update and log metrics
        self.train_metrics(y_hat, y)
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
        x, y = batch["image"], batch["target"]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        # Update and log metrics
        self.val_metrics(y_hat, y)
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
        x, y = batch["image"], batch["target"]
        y_hat = self(x)
        
        # Update and log metrics
        self.test_metrics(y_hat, y)
        self.log_dict(self.test_metrics)
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Select optimizer
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                momentum=0.9
            )
        elif self.hparams.optimizer == "rmsprop":
            optimizer = torch.optim.RMSprop(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer}")
        
        # Configure scheduler
        if self.hparams.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.get("t_max", 10),
                eta_min=1e-6
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        
        elif self.hparams.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=self.hparams.get("patience", 10),
                factor=self.hparams.get("factor", 0.1)
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        
        elif self.hparams.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.hparams.get("step_size", 30),
                gamma=self.hparams.get("gamma", 0.1)
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        
        elif self.hparams.scheduler == "none" or self.hparams.scheduler is None:
            return {"optimizer": optimizer}
        
        else:
            raise ValueError(f"Unsupported scheduler: {self.hparams.scheduler}")
    
    def visualize_batch(self, x, y, y_hat, stage, idx):
        """Visualize a batch of data.
        
        Args:
            x: Input images
            y: Target
            y_hat: Predictions
            stage: Current stage (train, val, test)
            idx: Current index/epoch
        """
        # Customize this method based on your task/data
        if not hasattr(self, "logger") or self.logger is None:
            return
            
        # Take the first few samples
        n_samples = min(4, x.size(0))
        
        fig, axs = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
        if n_samples == 1:
            axs = axs.reshape(1, -1)
            
        for i in range(n_samples):
            # Display input image (assumes first 3 channels are RGB)
            img = x[i, :3].permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)  # Ensure values are in [0, 1]
            axs[i, 0].imshow(img)
            axs[i, 0].set_title("Input")
            axs[i, 0].axis("off")
            
            # Display target (adapt this to your task)
            target = y[i, 0].cpu().numpy()
            axs[i, 1].imshow(target, cmap="viridis")
            axs[i, 1].set_title("Target")
            axs[i, 1].axis("off")
            
            # Display prediction
            pred = y_hat[i, 0].detach().cpu().numpy()
            axs[i, 2].imshow(pred, cmap="viridis")
            axs[i, 2].set_title("Prediction")
            axs[i, 2].axis("off")
        
        plt.tight_layout()
        
        # Log to tensorboard
        self.logger.experiment.add_figure(f"{stage}_visualization_epoch_{idx}", fig, global_step=self.global_step)
        plt.close(fig)


class SegmentationTask(BaseTask):
    """Task for semantic segmentation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add segmentation-specific metrics
        self.val_metrics.add_metrics({
            'val_precision': Precision(task="binary", threshold=0.5),
            'val_recall': Recall(task="binary", threshold=0.5),
            'val_f1': F1Score(task="binary", threshold=0.5)
        })
    
    def visualize_batch(self, x, y, y_hat, stage, idx):
        """Override to customize visualization for segmentation."""
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
            
            # Target mask (binary)
            target = y[i, 0].cpu().numpy()
            axs[i, 1].imshow(target, cmap="gray")
            axs[i, 1].set_title("Target Mask")
            axs[i, 1].axis("off")
            
            # Predicted mask
            pred = y_hat[i, 0].detach().cpu().numpy()
            pred = (pred > 0.5).astype(np.float32)  # Threshold
            axs[i, 2].imshow(pred, cmap="gray")
            axs[i, 2].set_title("Predicted Mask")
            axs[i, 2].axis("off")
        
        plt.tight_layout()
        self.logger.experiment.add_figure(f"{stage}_segmentation_{idx}", fig, global_step=self.global_step)
        plt.close(fig)


class RegressionTask(BaseTask):
    """Task for regression problems."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional regression metrics can be added here
    
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
        self.logger.experiment.add_figure(f"{stage}_regression_{idx}", fig, global_step=self.global_step)
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
    if task_type.lower() == "segmentation":
        return SegmentationTask(model, **kwargs)
    elif task_type.lower() == "regression":
        return RegressionTask(model, **kwargs)
    elif task_type.lower() == "base":
        return BaseTask(model, **kwargs)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
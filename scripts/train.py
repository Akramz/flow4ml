"""
Main training script for running ML model training.

This script:
1. Reads configuration from YAML file and/or command line args
2. Sets up model, data, and training components
3. Executes the training process or hyperparameter search
4. Logs results and saves checkpoints

The script supports two modes:
- Standard training: Train a model with fixed hyperparameters
- Search mode: Find optimal hyperparameters using Optuna

Usage:
    # Standard training
    python train.py --config configs/baseline/simple.yaml [additional args]
    
    # Hyperparameter search
    python train.py --config configs/baseline/simple.yaml --search_mode --n_trials 20 \
        --lr_range 1e-5,1e-2 --wd_range 1e-6,1e-3 --batch_size_range 16,32,64
"""

from pdb import set_trace
import argparse
import os
import yaml
from pathlib import Path
import time
import json

import torch
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

# Import from flow module
from flow.config import TrainerConfig
from flow.models import get_model
from flow.datamodules import get_datamodule
from flow.trainers import get_task

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a model")

    # Config file
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )

    # Optional overrides for config
    parser.add_argument(
        "--task_type",
        type=str,
        help="Task type (segmentation, classification, regression)",
    )
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument(
        "--backbone_name", type=str, help="Backbone name for applicable models"
    )
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--max_epochs", type=int, help="Maximum number of epochs")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--gpu_ids", type=int, nargs="+", help="GPU IDs to use")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--experiment_name", type=str, help="Experiment name")

    # Hyperparameter search mode
    parser.add_argument(
        "--search_mode",
        action="store_true",
        help="Enable Optuna hyperparameter search mode",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Number of trials for hyperparameter search",
    )
    parser.add_argument(
        "--search_epochs",
        type=int,
        default=10,
        help="Number of epochs per trial during search",
    )
    parser.add_argument(
        "--lr_range",
        type=str,
        default="1e-5,1e-2",
        help="Learning rate range for search (min,max). Example: 1e-5,1e-2",
    )
    parser.add_argument(
        "--wd_range",
        type=str,
        default="1e-6,1e-3",
        help="Weight decay range for search (min,max). Example: 1e-6,1e-3",
    )
    parser.add_argument(
        "--batch_size_range",
        type=str,
        default="16,32,64",
        help="Comma-separated list of batch sizes to try. Example: 16,32,64",
    )
    parser.add_argument(
        "--disable_tuning",
        type=str,
        nargs="+",
        help="List of hyperparameters to disable from tuning (e.g., --disable_tuning lr batch_size)",
        default=[],
    )
    parser.add_argument(
        "--frac",
        type=float,
        default=1.0,
        help="Fraction of data to sample for hyperparameter search (between 0 and 1)",
    )

    return parser.parse_args()


class HyperparameterSearch:
    """Class for performing hyperparameter optimization using Optuna."""

    def __init__(
        self,
        config_dict,
        task_type,
        n_trials=20,
        search_epochs=10,
        param_ranges=None,
        frac=1.0,
    ):
        """Initialize hyperparameter search.

        Args:
            config_dict: Configuration dictionary
            task_type: Type of task (segmentation, classification, etc.)
            n_trials: Number of trials to run
            search_epochs: Number of epochs to train per trial
            param_ranges: Dictionary of parameter ranges to search
            frac: Fraction of data to use for search
        """
        self.config_dict = config_dict.copy()
        self.task_type = task_type
        self.n_trials = n_trials
        self.search_epochs = search_epochs
        self.param_ranges = param_ranges or {}
        self.frac = frac

        # Override max_epochs for search
        self.config_dict["max_epochs"] = search_epochs

        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for hyperparameter search. Install with: pip install optuna"
            )

    def objective(self, trial):
        """Optuna objective function.

        Args:
            trial: Optuna trial object

        Returns:
            Validation loss
        """
        # Create a copy of the config for this trial
        trial_config = self.config_dict.copy()

        # Sample hyperparameters
        if "lr" in self.param_ranges:
            lr_min, lr_max = self.param_ranges["lr"]
            trial_config["lr"] = trial.suggest_float("lr", lr_min, lr_max, log=True)

        if "weight_decay" in self.param_ranges:
            wd_min, wd_max = self.param_ranges["weight_decay"]
            trial_config["weight_decay"] = trial.suggest_float(
                "weight_decay", wd_min, wd_max, log=True
            )

        if "batch_size" in self.param_ranges:
            batch_sizes = self.param_ranges["batch_size"]
            trial_config["batch_size"] = trial.suggest_categorical(
                "batch_size", batch_sizes
            )

        # Generate a unique experiment name for this trial
        trial_config["experiment_name"] = (
            f"{trial_config.get('experiment_short_name', 'search')}_trial_{trial.number}"
        )

        # Create config object
        config = TrainerConfig(**trial_config)

        # Initialize model, datamodule, and task
        model = get_model(config)
        datamodule = get_datamodule(self.task_type, config)
        task = get_task(
            model=model,
            **config.__dict__,
            scheduler_params={
                "patience": config.patience,
                "t_max": config.max_epochs // 2,
            },
        )

        # Setup callbacks
        early_stop = EarlyStopping(
            monitor="val_loss", patience=min(5, config.max_epochs // 2), mode="min"
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=config.output_dir,
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )

        # Setup logger
        logger = pl_loggers.TensorBoardLogger(
            save_dir=config.log_dir, name=config.experiment_name
        )

        # Train model
        trainer = pl.Trainer(
            max_epochs=config.max_epochs,
            accelerator="gpu" if config.gpu_ids else "cpu",
            devices=config.gpu_ids if config.gpu_ids else 1,
            logger=logger,
            callbacks=[early_stop, checkpoint_callback],
            enable_progress_bar=False,  # Disable progress bar for cleaner output
            precision=config.precision,
        )

        trainer.fit(model=task, datamodule=datamodule)

        # Return best validation loss
        return early_stop.best_score.item()

    def run(self):
        """Run hyperparameter search.

        Returns:
            Tuple of (best_params, best_score)
        """
        print(f"Starting hyperparameter search with {self.n_trials} trials")
        print(f"Parameter ranges: {self.param_ranges}")

        # Create study
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.n_trials)

        # Get best trial
        best_trial = study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value

        return best_params, best_value


def main():
    """Main training function."""
    # Parse args and read config
    args = parse_args()

    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    # Override config with command line args
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None and arg_name not in [
            "config",
            "search_mode",
            "n_trials",
            "lr_range",
            "wd_range",
            "batch_size_range",
            "disable_tuning",
            "frac",
            "search_epochs",
        ]:
            config_dict[arg_name] = arg_value

    # Get task type
    task_type = config_dict.get("task_type", "base")

    # Check if running in search mode
    if args.search_mode:
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for hyperparameter search. Install with: pip install optuna"
            )

        print("Running in hyperparameter search mode")

        # Parse parameter ranges
        param_ranges = {}

        # Learning rate range
        if "lr" not in args.disable_tuning:
            try:
                lr_min, lr_max = map(float, args.lr_range.split(","))
                param_ranges["lr"] = (lr_min, lr_max)
            except ValueError:
                print(f"Invalid lr_range format: {args.lr_range}. Using default.")
                param_ranges["lr"] = (1e-5, 1e-2)

        # Weight decay range
        if "weight_decay" not in args.disable_tuning:
            try:
                wd_min, wd_max = map(float, args.wd_range.split(","))
                param_ranges["weight_decay"] = (wd_min, wd_max)
            except ValueError:
                print(f"Invalid wd_range format: {args.wd_range}. Using default.")
                param_ranges["weight_decay"] = (1e-6, 1e-3)

        # Batch size options
        if "batch_size" not in args.disable_tuning:
            try:
                batch_sizes = [int(bs) for bs in args.batch_size_range.split(",")]
                param_ranges["batch_size"] = batch_sizes
            except ValueError:
                print(
                    f"Invalid batch_size_range format: {args.batch_size_range}. Using default."
                )
                param_ranges["batch_size"] = [16, 32, 64]

        # Run hyperparameter search
        search = HyperparameterSearch(
            config_dict=config_dict,
            task_type=task_type,
            n_trials=args.n_trials,
            search_epochs=args.search_epochs,
            param_ranges=param_ranges,
            frac=args.frac,
        )

        start_time = time.time()
        best_params, best_score = search.run()
        search_time = time.time() - start_time

        # Print results
        print("\nHyperparameter search completed!")
        print(f"Time taken: {search_time:.2f} seconds")
        print("\nBest hyperparameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"Best validation loss: {best_score:.6f}")

        # Save results to file
        results = {
            "best_params": best_params,
            "best_score": best_score,
            "search_time": search_time,
            "n_trials": args.n_trials,
            "param_ranges": param_ranges,
        }

        # Update config with best parameters
        for param, value in best_params.items():
            config_dict[param] = value

        # Save best parameters to file
        os.makedirs("search_results", exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_file = f"search_results/search_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {results_file}")

        # Ask if user wants to train with best parameters
        while True:
            choice = input(
                "\nDo you want to train a model with these parameters? (y/n): "
            ).lower()
            if choice in ["y", "yes"]:
                break
            elif choice in ["n", "no"]:
                return
            else:
                print("Invalid choice. Please enter 'y' or 'n'.")

    # Create config object for validation
    config = TrainerConfig(**config_dict)

    # Print configuration summary
    print(f"\nExperiment: {config.experiment_name}")
    print(f"Task type: {task_type}")
    print(f"Model: {config.model_name}")
    print(f"Backbone: {config.backbone_name}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.lr}")
    print(f"Weight decay: {config.weight_decay}")
    print(f"Max epochs: {config.max_epochs}")
    print(f"GPUs: {config.gpu_ids}")

    # Set random seed
    pl.seed_everything(config.seed)

    # Get model
    model = get_model(config)

    # Get datamodule
    datamodule = get_datamodule(task_type, config)

    # Get task
    task = get_task(
        model=model,
        **config.__dict__,
        scheduler_params={
            "scheduler": config.scheduler,
            "patience": config.patience,
            "t_max": config.max_epochs // 10,
        },
    )

    # Setup callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.output_dir,
        filename="{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_epochs=getattr(config, "checkpoint_every", 10),
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    if getattr(config, "early_stop", True):
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=getattr(config, "early_stop_patience", 20),
            mode="min",
        )
        callbacks.append(early_stop_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    # Setup logger
    logger = pl_loggers.TensorBoardLogger(
        save_dir=config.log_dir, name=config.experiment_name
    )

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="gpu" if config.gpu_ids else "cpu",
        devices=config.gpu_ids if config.gpu_ids else 1,
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
        precision=config.precision,
    )

    # Train model
    trainer.fit(
        model=task,
        datamodule=datamodule,
        ckpt_path=getattr(config, "resume_from_checkpoint", None),
    )

    # Print results
    print(f"Best model path: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score}")

    # Save best model path
    with open(os.path.join(config.output_dir, "best_model.txt"), "w") as f:
        f.write(f"Best model path: {checkpoint_callback.best_model_path}\n")
        f.write(f"Best validation loss: {checkpoint_callback.best_model_score}\n")


if __name__ == "__main__":
    main()

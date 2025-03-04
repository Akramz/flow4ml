"""
Evaluation script for model assessment.

This script:
1. Loads a trained model from a checkpoint
2. Runs inference on test data
3. Computes and reports metrics
4. Generates visualizations for error analysis

Usage:
    python evaluate.py --model_path model_runs/experiment/best.ckpt --test_data path/to/test/data
"""

import argparse
import os
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from lightning.pytorch import Trainer
import yaml

from flow.config import TrainerConfig
from flow.models import get_model
from flow.datamodules import get_datamodule
from flow.trainers import get_task


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained model")

    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model checkpoint"
    )

    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to test data (directory or CSV file)",
    )

    parser.add_argument(
        "--config", type=str, help="Path to original config file (optional)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results",
    )

    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )

    parser.add_argument(
        "--save_predictions", action="store_true", help="Save model predictions to disk"
    )

    parser.add_argument(
        "--task_type",
        type=str,
        default="base",
        help="Task type (segmentation, classification, regression)",
    )

    parser.add_argument(
        "--gpu_id", type=int, default=0, help="GPU ID to use for evaluation"
    )

    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load configuration
    if args.config:
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)
    else:
        # If no config provided, load from checkpoint hparams
        checkpoint = torch.load(args.model_path, map_location="cpu")
        if "hyper_parameters" in checkpoint:
            config_dict = checkpoint["hyper_parameters"]
        else:
            print(
                "No config file provided and no hyper_parameters found in checkpoint."
            )
            print("Using default configuration.")
            config_dict = {}

    # Override with evaluation-specific settings
    config_dict["input_dirs"] = [args.test_data]
    config_dict["batch_size"] = args.batch_size

    # Create config object
    config = TrainerConfig(**config_dict)

    # Load model
    print(f"Loading model from {args.model_path}")
    model = get_model(config)

    # Create task
    task = get_task(args.task_type, model)

    # Load checkpoint weights
    checkpoint = torch.load(args.model_path, map_location="cpu")
    task.load_state_dict(checkpoint["state_dict"])

    # Load data
    datamodule = get_datamodule(args.task_type, config)
    datamodule.setup(stage="test")

    # Create trainer
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[args.gpu_id] if torch.cuda.is_available() else None,
        logger=None,
        enable_checkpointing=False,
    )

    # Run evaluation
    print("Running evaluation...")
    results = trainer.test(task, datamodule=datamodule)[0]

    # Print results
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

    # Save results to file
    result_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {result_file}")

    # Save predictions if requested
    if args.save_predictions:
        print("Generating and saving predictions...")
        save_predictions(task, datamodule, args.output_dir)


def save_predictions(task, datamodule, output_dir):
    """Generate and save model predictions.

    Args:
        task: LightningModule task
        datamodule: DataModule with test data
        output_dir: Directory to save predictions
    """
    # Create predictions directory
    predictions_dir = os.path.join(output_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    # Set model to evaluation mode
    task.eval()

    # Generate predictions
    dataloader = datamodule.test_dataloader()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # Get input and generate prediction
            x = batch["image"]
            y_hat = task(x)

            # Convert to numpy
            pred = y_hat.cpu().numpy()

            # Save first 10 batches
            if i < 10:
                for j in range(min(4, len(x))):
                    # Create figure
                    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

                    # Plot input image
                    img = x[j, :3].permute(1, 2, 0).cpu().numpy()
                    img = np.clip(img, 0, 1)
                    ax[0].imshow(img)
                    ax[0].set_title("Input")
                    ax[0].axis("off")

                    # Plot ground truth if available
                    if "target" in batch:
                        target = batch["target"][j, 0].cpu().numpy()
                        ax[1].imshow(target, cmap="viridis")
                        ax[1].set_title("Ground Truth")
                        ax[1].axis("off")
                    else:
                        ax[1].axis("off")

                    # Plot prediction
                    ax[2].imshow(pred[j, 0], cmap="viridis")
                    ax[2].set_title("Prediction")
                    ax[2].axis("off")

                    # Save figure
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(predictions_dir, f"pred_batch{i}_sample{j}.png")
                    )
                    plt.close()

    print(f"Saved prediction visualizations to {predictions_dir}")


if __name__ == "__main__":
    main()

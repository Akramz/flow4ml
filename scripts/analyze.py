"""
Error analysis script for model improvement.

This script:
1. Loads a trained model and test data
2. Runs inference and computes errors
3. Analyzes patterns in errors
4. Generates visualizations highlighting problematic cases
5. Suggests potential improvements

Usage:
    python analyze.py --model_path model_runs/experiment/best.ckpt --test_data path/to/test/data
"""

import argparse
import os
import json
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import yaml

from flow.config import TrainerConfig
from flow.models import get_model
from flow.datamodules import get_datamodule
from flow.trainers import get_task


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze model errors")

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
        default="analysis_results",
        help="Directory to save analysis results",
    )

    parser.add_argument(
        "--n_worst_samples",
        type=int,
        default=20,
        help="Number of worst samples to visualize",
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


def compute_errors(task, dataloader):
    """Compute errors for all samples in dataloader.

    Args:
        task: LightningModule task
        dataloader: Test dataloader

    Returns:
        Dictionary with error information
    """
    task.eval()
    device = next(task.parameters()).device

    all_errors = []
    all_metadata = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing errors"):
            # Get input and target
            x = batch["image"].to(device)

            if "target" not in batch:
                continue

            y = batch["target"].to(device)

            # Generate prediction
            y_hat = task(x)

            # Compute error for each sample
            for i in range(len(x)):
                # Mean absolute error for this sample
                mae = torch.abs(y_hat[i] - y[i]).mean().item()

                # Max error for this sample
                max_error = torch.abs(y_hat[i] - y[i]).max().item()

                # Sample metadata
                metadata = {
                    key: batch[key][i]
                    for key in batch
                    if key not in ["image", "target"] and isinstance(batch[key], list)
                }

                # Store results
                all_errors.append(
                    {
                        "mae": mae,
                        "max_error": max_error,
                        "input": x[i].cpu(),
                        "target": y[i].cpu(),
                        "prediction": y_hat[i].cpu(),
                    }
                )
                all_metadata.append(metadata)

    return {"errors": all_errors, "metadata": all_metadata}


def analyze_errors(error_data, n_worst=20):
    """Analyze error patterns.

    Args:
        error_data: Dictionary with error information
        n_worst: Number of worst samples to select

    Returns:
        Analysis results
    """
    errors = error_data["errors"]
    metadata = error_data["metadata"]

    # Sort samples by error
    sorted_indices = sorted(
        range(len(errors)), key=lambda i: errors[i]["mae"], reverse=True
    )

    # Select worst samples
    worst_indices = sorted_indices[:n_worst]
    worst_samples = [errors[i] for i in worst_indices]
    worst_metadata = [metadata[i] for i in worst_indices]

    # Compute error statistics
    all_maes = [e["mae"] for e in errors]
    all_max_errors = [e["max_error"] for e in errors]

    stats = {
        "mean_mae": np.mean(all_maes),
        "median_mae": np.median(all_maes),
        "std_mae": np.std(all_maes),
        "max_mae": np.max(all_maes),
        "mean_max_error": np.mean(all_max_errors),
        "median_max_error": np.median(all_max_errors),
        "worst_samples": worst_samples,
        "worst_metadata": worst_metadata,
    }

    # Metadata analysis if available
    if metadata and len(metadata) > 0 and len(metadata[0]) > 0:
        # Convert metadata to DataFrame for analysis
        df = pd.DataFrame(metadata)

        # Add error values
        df["mae"] = all_maes
        df["max_error"] = all_max_errors

        # Analyze correlations between metadata and errors
        stats["metadata_analysis"] = df

    return stats


def visualize_worst_samples(analysis_results, output_dir):
    """Visualize worst samples.

    Args:
        analysis_results: Analysis results from analyze_errors
        output_dir: Directory to save visualizations
    """
    worst_samples = analysis_results["worst_samples"]
    worst_metadata = analysis_results["worst_metadata"]

    # Create output directory
    worst_dir = os.path.join(output_dir, "worst_samples")
    os.makedirs(worst_dir, exist_ok=True)

    # Visualize each sample
    for i, sample in enumerate(worst_samples):
        # Create figure
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Input image
        img = sample["input"][:3].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[0].imshow(img)
        axes[0].set_title("Input")
        axes[0].axis("off")

        # Target
        target = sample["target"][0].numpy()
        im = axes[1].imshow(target, cmap="viridis")
        axes[1].set_title("Target")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1])

        # Prediction
        pred = sample["prediction"][0].numpy()
        im = axes[2].imshow(pred, cmap="viridis")
        axes[2].set_title("Prediction")
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2])

        # Error map
        error_map = np.abs(pred - target)
        im = axes[3].imshow(error_map, cmap="hot")
        axes[3].set_title(f"Error (MAE: {sample['mae']:.4f})")
        axes[3].axis("off")
        plt.colorbar(im, ax=axes[3])

        # Add metadata as text if available
        if worst_metadata and len(worst_metadata) > i:
            metadata = worst_metadata[i]
            metadata_str = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
            plt.figtext(
                0.5,
                0.01,
                metadata_str,
                ha="center",
                fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.5, "pad": 5},
            )

        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(worst_dir, f"worst_sample_{i+1}.png"))
        plt.close()


def create_error_histograms(analysis_results, output_dir):
    """Create histograms of errors.

    Args:
        analysis_results: Analysis results from analyze_errors
        output_dir: Directory to save visualizations
    """
    if "metadata_analysis" not in analysis_results:
        return

    df = analysis_results["metadata_analysis"]

    # Create output directory
    hist_dir = os.path.join(output_dir, "histograms")
    os.makedirs(hist_dir, exist_ok=True)

    # Create histogram of errors
    plt.figure(figsize=(10, 6))
    sns.histplot(df["mae"], kde=True)
    plt.title("Distribution of Mean Absolute Errors")
    plt.xlabel("MAE")
    plt.ylabel("Count")
    plt.axvline(
        df["mae"].mean(),
        color="r",
        linestyle="--",
        label=f"Mean: {df['mae'].mean():.4f}",
    )
    plt.axvline(
        df["mae"].median(),
        color="g",
        linestyle="--",
        label=f"Median: {df['mae'].median():.4f}",
    )
    plt.legend()
    plt.savefig(os.path.join(hist_dir, "mae_histogram.png"))
    plt.close()

    # Analysis by metadata categories if available
    for col in df.columns:
        if col in ["mae", "max_error"]:
            continue

        # Check if column has categorical data
        if (
            df[col].nunique() < 20
        ):  # Only for categorical data with limited unique values
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=col, y="mae", data=df)
            plt.title(f"MAE by {col}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(hist_dir, f"mae_by_{col}.png"))
            plt.close()


def main():
    """Main analysis function."""
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

    # Move to device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    task = task.to(device)

    # Load data
    datamodule = get_datamodule(args.task_type, config)
    datamodule.setup(stage="test")
    test_dataloader = datamodule.test_dataloader()

    # Compute errors
    print("Computing errors...")
    error_data = compute_errors(task, test_dataloader)

    # Analyze errors
    print("Analyzing error patterns...")
    analysis_results = analyze_errors(error_data, n_worst=args.n_worst_samples)

    # Print error statistics
    print("\nError Statistics:")
    print(f"  Mean MAE: {analysis_results['mean_mae']:.4f}")
    print(f"  Median MAE: {analysis_results['median_mae']:.4f}")
    print(f"  Std MAE: {analysis_results['std_mae']:.4f}")
    print(f"  Max MAE: {analysis_results['max_mae']:.4f}")
    print(f"  Mean Max Error: {analysis_results['mean_max_error']:.4f}")

    # Save error statistics
    stats_file = os.path.join(args.output_dir, "error_statistics.json")
    with open(stats_file, "w") as f:
        # Convert non-serializable parts to lists or basic types
        serializable_stats = {
            "mean_mae": float(analysis_results["mean_mae"]),
            "median_mae": float(analysis_results["median_mae"]),
            "std_mae": float(analysis_results["std_mae"]),
            "max_mae": float(analysis_results["max_mae"]),
            "mean_max_error": float(analysis_results["mean_max_error"]),
            "median_max_error": float(analysis_results["median_max_error"]),
        }
        json.dump(serializable_stats, f, indent=2)

    # Visualize worst samples
    print("Visualizing worst samples...")
    visualize_worst_samples(analysis_results, args.output_dir)

    # Create error histograms
    print("Creating error histograms...")
    create_error_histograms(analysis_results, args.output_dir)

    print(f"Analysis complete! Results saved to {args.output_dir}")
    print("\nNext steps to improve model performance:")
    print("1. Review visualizations of worst samples to identify patterns")
    print("2. Check if certain metadata categories correlate with higher errors")
    print("3. Consider model architecture changes or targeted augmentations")
    print("4. Experiment with different loss functions or regularization")


if __name__ == "__main__":
    main()

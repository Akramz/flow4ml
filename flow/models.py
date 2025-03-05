"""
Models module for implementing custom model architectures.

This file provides:
1. Implementation of a random baseline model for benchmarking
2. Custom model implementations
3. Factory function for model creation

For TorchGeo models (UNet, UPerNet, DeepLabV3, etc.), we use the
TorchGeo trainer directly which handles model creation.
"""

import torch
import torch.nn as nn


class RandomModel(nn.Module):
    """A model that outputs random predictions.

    This can be useful for testing data pipelines and evaluation metrics
    without implementing actual model logic.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        **kwargs: Additional model parameters
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        """Initialize the random model.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            **kwargs: Additional model parameters
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.random_weight = nn.Parameter(torch.randn(1))
        self.random_bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        """Forward pass that returns random predictions with the same shape as expected output.

        Args:
            x: Input tensor of shape [batch_size, in_channels, height, width]

        Returns:
            Random tensor of shape [batch_size, out_channels, height, width]
        """
        batch_size, _, height, width = x.shape
        random_noise = torch.rand(
            batch_size, self.out_channels, height, width, device=x.device
        )
        return random_noise * self.random_weight + self.random_bias


class CustomModel(nn.Module):
    """Example custom model. Replace with your implementation."""

    def __init__(self, config):
        """Initialize your model.

        Args:
            config: Configuration object containing model parameters
        """
        super().__init__()
        # TODO: Initialize your model architecture
        self.example_conv = nn.Conv2d(
            in_channels=config.in_channels, out_channels=64, kernel_size=3, padding=1
        )
        # Add more layers as needed

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Model output
        """
        # TODO: Implement forward pass
        x = self.example_conv(x)
        # Process through other layers
        return x


def get_model(config):
    """Factory function to create a model instance from config.

    Args:
        config: Configuration object with model parameters

    Returns:
        A model instance
    """
    # Random baseline model is handled here,
    # other models (unet, upernet, etc.) are handled by TorchGeo trainers
    if config.model_name == "random":
        return RandomModel(
            in_channels=config.in_channels, out_channels=config.out_channels
        )
    elif config.model_name == "custom":
        return CustomModel(config)
    else:
        # For TorchGeo models, return None and let the trainer handle model creation
        return None

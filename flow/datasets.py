"""
Dataset utilities for loading and preprocessing data.

This module provides base dataset classes and utility functions for:
- Loading image data from disk or remote sources
- Preprocessing and augmenting data
- Converting between different data formats

Customize these classes for your specific data structure and tasks.
"""

import os
from typing import Dict, List, Callable, Optional, Tuple, Union
from pdb import set_trace

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import pandas as pd


def stack_samples(samples):
    """Stack a list of samples into a batch.

    Args:
        samples: List of sample dictionaries

    Returns:
        Dictionary with stacked tensors
    """
    batch = {}
    for key in samples[0].keys():
        if isinstance(samples[0][key], torch.Tensor):
            batch[key] = torch.stack([s[key] for s in samples])
        else:
            batch[key] = [s[key] for s in samples]
    return batch


class BaseImageDataset(Dataset):
    """Base dataset for loading images and targets."""

    def __init__(
        self,
        image_paths: List[str],
        target_paths: Optional[List[str]] = None,
        transforms: Optional[Callable] = None,
        image_size: Tuple[int, int] = (224, 224),
    ):
        """Initialize dataset.

        Args:
            image_paths: List of paths to images
            target_paths: List of paths to targets (optional)
            transforms: Transforms to apply
            image_size: Size to resize images to
        """
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transforms = transforms
        self.image_size = image_size

        # Default transforms if none provided
        if self.transforms is None:
            self.transforms = T.Compose(
                [
                    T.Resize(image_size),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Get dataset item.

        Args:
            idx: Index

        Returns:
            Dictionary with image and target
        """
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # Apply transforms
        image = self.transforms(image)

        # Create sample dictionary
        sample = {"image": image, "path": img_path}

        # Load target if available
        if self.target_paths is not None:
            target_path = self.target_paths[idx]
            # This is a placeholder - modify based on your target format
            # (e.g., segmentation mask, regression values, classification labels)
            target = Image.open(target_path).convert("L")
            target = T.Resize(self.image_size)(target)
            target = T.ToTensor()(target)
            sample["mask"] = target

        return sample


class CSVDataset(BaseImageDataset):
    """Dataset loading data paths from a CSV file."""

    def __init__(
        self,
        csv_path: str,
        image_col: str = "image_path",
        target_col: Optional[str] = "target_path",
        transforms: Optional[Callable] = None,
        image_size: Tuple[int, int] = (224, 224),
        root_dir: Optional[str] = None,
    ):
        """Initialize dataset.

        Args:
            csv_path: Path to CSV file
            image_col: Column name for image paths
            target_col: Column name for target paths
            transforms: Transforms to apply
            image_size: Size to resize images to
            root_dir: Root directory to prepend to relative paths
        """
        # Load CSV
        self.df = pd.read_csv(csv_path)

        # Validate columns
        if image_col not in self.df.columns:
            raise ValueError(f"CSV file does not contain column '{image_col}'")

        if target_col is not None and target_col not in self.df.columns:
            print(
                f"Warning: CSV file does not contain target column '{target_col}'. Running without targets."
            )
            target_col = None

        # Extract paths
        image_paths = self.df[image_col].tolist()
        target_paths = None if target_col is None else self.df[target_col].tolist()

        # Prepend root directory if provided
        if root_dir is not None:
            image_paths = [os.path.join(root_dir, p) for p in image_paths]
            if target_paths is not None:
                target_paths = [os.path.join(root_dir, p) for p in target_paths]

        # Initialize base class
        super().__init__(image_paths, target_paths, transforms, image_size)

        # Store additional metadata
        self.metadata = {
            col: self.df[col].tolist()
            for col in self.df.columns
            if col not in [image_col, target_col]
        }

    def __getitem__(self, idx):
        """Get dataset item with metadata."""
        sample = super().__getitem__(idx)

        # Add metadata
        for key, values in self.metadata.items():
            sample[key] = values[idx]

        return sample


# Add more dataset classes as needed for your specific task
# Example: segmentation, classification, remote sensing, etc.


class EarthSurfaceWaterDataset:
    """Dataset for the Earth Surface Water dataset from TorchGeo."""

    def scale(self, item):
        """Scale Sentinel-2 values to reflectance.

        Args:
            item: Item to scale

        Returns:
            Scaled item
        """
        item["image"] = item["image"] / 10000
        return item

    def __init__(
        self,
        root_dir=None,
        dataset_url="https://hf.co/datasets/cordmaur/earth_surface_water/resolve/main/earth_surface_water.zip",
        transform=None,
        image_size=(512, 512),
        normalize=True,
        spectral_indices=None,
    ):
        """Initialize the Earth Surface Water dataset.

        Args:
            root_dir: Root directory where the dataset will be stored
            dataset_url: URL to download the dataset from
            transform: Transforms to apply to the data
            image_size: Image size (height, width)
            normalize: Whether to normalize the data
            spectral_indices: List of spectral indices to compute
        """
        import tempfile
        from pathlib import Path
        import torch

        # Import TorchGeo components
        try:
            from torchgeo.datasets import (
                RasterDataset,
                utils,
                unbind_samples,
                stack_samples,
            )
            from torchgeo.samplers import RandomGeoSampler, Units
            from torchgeo.transforms import indices
        except ImportError:
            raise ImportError(
                "TorchGeo is required for the Earth Surface Water dataset"
            )

        self.root_dir = root_dir or Path(tempfile.gettempdir()) / "surface_water"
        self.dataset_url = dataset_url
        self.image_size = image_size
        self.normalize = normalize
        self.spectral_indices = spectral_indices

        # Download and extract dataset if needed
        utils.download_and_extract_archive(
            self.dataset_url,
            self.root_dir,
        )

        # Define the root path to the extracted dataset
        extracted_path = self.root_dir / "dset-s2"

        # Create the training datasets
        self.train_imgs = RasterDataset(
            paths=(extracted_path / "tra_scene").as_posix(),
            crs="epsg:3395",
            res=10,
            transforms=self.scale,
        )
        self.train_msks = RasterDataset(
            paths=(extracted_path / "tra_truth").as_posix(), crs="epsg:3395", res=10
        )

        # Create the validation datasets
        self.valid_imgs = RasterDataset(
            paths=(extracted_path / "val_scene").as_posix(),
            crs="epsg:3395",
            res=10,
            transforms=self.scale,
        )
        self.valid_msks = RasterDataset(
            paths=(extracted_path / "val_truth").as_posix(), crs="epsg:3395", res=10
        )

        # Mark the masks as non-images
        self.train_msks.is_image = False
        self.valid_msks.is_image = False

        # Combine images with masks
        self.train_dset = self.train_imgs & self.train_msks
        self.valid_dset = self.valid_imgs & self.valid_msks

        # Create the samplers
        self.train_sampler = RandomGeoSampler(
            self.train_imgs, size=self.image_size[0], length=130, units=Units.PIXELS
        )
        self.valid_sampler = RandomGeoSampler(
            self.valid_imgs, size=self.image_size[0], length=64, units=Units.PIXELS
        )

        # Calculate dataset statistics
        if self.normalize:
            mean, std = self.calc_statistics(self.train_imgs)
            # For spectral indices, append zeros/ones to mean/std
            if self.spectral_indices:
                import numpy as np

                n_indices = len(self.spectral_indices)
                mean = np.concatenate([mean, np.zeros(n_indices)])
                std = np.concatenate([std, np.ones(n_indices)])
            self.mean = mean
            self.std = std

        # Generate transforms
        self.transform = self.create_transforms()

    def calc_statistics(self, dataset):
        """Calculate mean and standard deviation for the dataset.

        Args:
            dataset: Dataset to calculate statistics for

        Returns:
            Mean and standard deviation (numpy arrays)
        """
        import numpy as np
        import rasterio as rio

        # Get files from the dataset's index
        files = [
            item.object
            for item in dataset.index.intersection(dataset.index.bounds, objects=True)
        ]

        # Reset statistics
        accum_mean = 0
        accum_std = 0

        # Calculate statistics for each file
        for file in files:
            img = rio.open(file).read() / 10000
            accum_mean += img.reshape((img.shape[0], -1)).mean(axis=1)
            accum_std += img.reshape((img.shape[0], -1)).std(axis=1)

        return accum_mean / len(files), accum_std / len(files)

    def create_transforms(self):
        """Create transforms for the dataset.

        Returns:
            Transforms to apply to the data
        """
        import torch
        import torch.nn as nn
        import kornia.augmentation as K
        from torchgeo.transforms import indices

        transforms = []

        # Add spectral indices
        if self.spectral_indices:
            for idx in self.spectral_indices:
                if idx.type == "NDWI":
                    transforms.append(
                        indices.AppendNDWI(
                            index_green=idx.index_green,
                            index_nir=idx.index_nir,
                        )
                    )
                elif idx.type == "NDVI":
                    transforms.append(
                        indices.AppendNDVI(
                            index_nir=idx.index_nir,
                            index_red=idx.index_red,
                        )
                    )

        # Add normalization
        if self.normalize:
            transforms.append(K.Normalize(mean=self.mean, std=self.std))

        # Return the sequential transform
        return nn.Sequential(*transforms) if transforms else None

    def __len__(self):
        """Return the total number of samples in the dataset.

        Returns:
            int: Total number of samples (train + validation)
        """
        # Return the combined length of training and validation samplers
        return len(self.train_sampler) + len(self.valid_sampler)

    def __getitem__(self, idx):
        """Get a random sample from the dataset.

        Args:
            idx: Index is ignored since we're using a random sampler

        Returns:
            Sample dictionary with image and mask tensors
        """
        # Import torch
        import torch

        # Determine whether to use training or validation dataset based on idx
        # For reproducibility, we'll use a seed based on idx, but still get random samples
        train_ratio = len(self.train_sampler) / (
            len(self.train_sampler) + len(self.valid_sampler)
        )
        use_train = torch.rand(1).item() < train_ratio

        # Set a seed based on idx for reproducibility
        torch.manual_seed(idx)

        try:
            if use_train:
                # Get a random sample from training dataset
                coords = next(iter(self.train_sampler))
                sample_dict = self.train_dset[coords]
            else:
                # Get a random sample from validation dataset
                coords = next(iter(self.valid_sampler))
                sample_dict = self.valid_dset[coords]

            # Extract image and mask tensors directly from the sample
            image = sample_dict["image"]
            mask = sample_dict["mask"].float()

            # Apply transforms if available
            if self.transform is not None:
                image = self.transform(image)

            return {"image": image.squeeze(), "mask": mask.squeeze().long()}

        except Exception as e:
            # If there's an error, return a dummy sample
            print(f"Error getting sample: {e}")
            print(f"Returning dummy sample for index {idx}")

            # Create dummy sample with proper dimensions
            image_shape = (3, self.image_size[0], self.image_size[1])
            mask_shape = (1, self.image_size[0], self.image_size[1])

            return {"image": torch.zeros(image_shape), "mask": torch.zeros(mask_shape)}


def get_dataset(config, transform=None):
    """Get dataset based on configuration.

    Args:
        config: Configuration object
        transform: Optional transforms to apply

    Returns:
        Dataset instance
    """
    # Check for Earth Surface Water dataset
    if hasattr(config, "dataset") and config.dataset == "earth_surface_water":
        # Handle Earth Surface Water dataset
        spectral_indices = getattr(config, "spectral_indices", None)
        normalize = getattr(config, "normalize", True)
        image_size = getattr(config, "image_size", (512, 512))
        dataset_url = getattr(
            config,
            "dataset_url",
            "https://hf.co/datasets/cordmaur/earth_surface_water/resolve/main/earth_surface_water.zip",
        )

        return EarthSurfaceWaterDataset(
            dataset_url=dataset_url,
            image_size=image_size,
            normalize=normalize,
            spectral_indices=spectral_indices,
        )

    # Handle other dataset types
    input_dirs = config.input_dirs
    if isinstance(input_dirs, str):
        input_dirs = [input_dirs]

    if len(input_dirs) == 0:
        raise ValueError("No input directories specified")

    first_input = input_dirs[0]
    if first_input.endswith(".csv"):
        # CSV dataset
        return CSVDataset(
            csv_path=first_input,
            image_col=getattr(config, "image_col", "image_path"),
            target_col=getattr(config, "target_col", "target_path"),
            transforms=transform,
            image_size=getattr(config, "image_size", (224, 224)),
            root_dir=getattr(config, "root_dir", None),
        )
    else:
        # Directory dataset
        image_paths = []
        target_paths = []

        # This is just a simple implementation - customize as needed
        for directory in input_dirs:
            if not os.path.exists(directory):
                continue

            # Find all images with common extensions
            for ext in ["jpg", "jpeg", "png", "tif", "tiff"]:
                image_paths.extend(
                    [
                        os.path.join(directory, f)
                        for f in os.listdir(directory)
                        if f.lower().endswith(f".{ext}")
                    ]
                )

        return BaseImageDataset(
            image_paths=image_paths,
            target_paths=None,  # Customize target finding logic
            transforms=transform,
            image_size=getattr(config, "image_size", (224, 224)),
        )

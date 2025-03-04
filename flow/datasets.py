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
            self.transforms = T.Compose([
                T.Resize(image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
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
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        image = self.transforms(image)
        
        # Create sample dictionary
        sample = {"image": image, "path": img_path}
        
        # Load target if available
        if self.target_paths is not None:
            target_path = self.target_paths[idx]
            # This is a placeholder - modify based on your target format
            # (e.g., segmentation mask, regression values, classification labels)
            target = Image.open(target_path).convert('L')  
            target = T.Resize(self.image_size)(target)
            target = T.ToTensor()(target)
            sample["target"] = target
        
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
            print(f"Warning: CSV file does not contain target column '{target_col}'. Running without targets.")
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
        self.metadata = {col: self.df[col].tolist() for col in self.df.columns 
                         if col not in [image_col, target_col]}
    
    def __getitem__(self, idx):
        """Get dataset item with metadata."""
        sample = super().__getitem__(idx)
        
        # Add metadata
        for key, values in self.metadata.items():
            sample[key] = values[idx]
        
        return sample


# Add more dataset classes as needed for your specific task
# Example: segmentation, classification, remote sensing, etc.


def get_dataset(config, transform=None):
    """Get dataset based on configuration.
    
    Args:
        config: Configuration object
        transform: Optional transforms to apply
        
    Returns:
        Dataset instance
    """
    # Detect dataset type from input dirs
    input_dirs = config.input_dirs
    if isinstance(input_dirs, str):
        input_dirs = [input_dirs]
    
    if len(input_dirs) == 0:
        raise ValueError("No input directories specified")
    
    first_input = input_dirs[0]
    if first_input.endswith('.csv'):
        # CSV dataset
        return CSVDataset(
            csv_path=first_input,
            image_col=getattr(config, "image_col", "image_path"),
            target_col=getattr(config, "target_col", "target_path"),
            transforms=transform,
            image_size=getattr(config, "image_size", (224, 224)),
            root_dir=getattr(config, "root_dir", None)
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
            for ext in ['jpg', 'jpeg', 'png', 'tif', 'tiff']:
                image_paths.extend(
                    [os.path.join(directory, f) for f in os.listdir(directory) 
                     if f.lower().endswith(f'.{ext}')]
                )
        
        return BaseImageDataset(
            image_paths=image_paths,
            target_paths=None,  # Customize target finding logic
            transforms=transform,
            image_size=getattr(config, "image_size", (224, 224))
        )
# Configuration Files

This directory contains YAML configuration files for model training. Each file defines a unique experiment setup, including model architecture, hyperparameters, and training parameters.

## Directory Structure

Configuration files are organized by research direction, with each sub-directory representing a specific branch of investigation:

```
configs/
├── 0_baselines/         # Initial baseline experiments
├── 1_architectures/     # Testing different model architectures
├── 2_learning_rates/    # Optimizing learning rate schedules
├── 3_data_augmentation/ # Exploring augmentation strategies
└── ...                  # Additional research directions
```

This structure:
- Aligns with Git branches for each research direction
- Creates a clear experimental sequence
- Makes it easier to track progress across different approaches
- Prevents overcrowding a single experiments directory

## Configuration File Format

Configuration files are YAML format and should include these key sections:

```yaml
# Experiment metadata
task_type: "base"  # base, segmentation, classification, regression
experiment_short_name: "experiment_description"

# Model parameters
model_name: "unet"  # Model architecture
backbone_name: "resnet18"  # Backbone network
weight_init: "pretrained"  # Weight initialization

# Input/output configuration
in_channels: 3  # Number of input channels
out_channels: 1  # Number of output channels
image_size: [224, 224]  # Input image size

# Optimizer parameters
optimizer: "adamw"  # Optimizer type
lr: 0.001  # Learning rate
weight_decay: 0.01  # Weight decay
scheduler: "cosine"  # LR scheduler

# Loss function
loss: "mse"  # Loss function

# Training parameters
batch_size: 32  # Batch size
max_epochs: 100  # Maximum training epochs
seed: 42  # Random seed
gpu_ids: [0]  # GPU IDs to use
```

## Creating New Experiments

When creating a new experiment:

1. **Identify the research direction**: Place the experiment in the appropriate sub-directory
2. **Start from an existing configuration**: Copy a relevant config file as your starting point
3. **Change only what's needed**: Modify only the parameters you want to test
4. **Use descriptive filenames**: Name files clearly to indicate what's being tested
5. **Organize in sequences**: Use numerical prefixes within each directory

## Experiment Naming Convention

Within each research direction, use a clear naming convention for experiment configs:

```
0_baselines/
├── 0_simple_baseline.yaml
├── 1_improved_baseline.yaml
└── 2_optimized_baseline.yaml

1_architectures/
├── 0_resnet18.yaml
├── 1_resnet50.yaml
└── 2_efficientnet.yaml
```

This sequential naming helps track the evolution of your experiments and highlights the specific changes being tested within each research direction.

## Scientific vs. Nuisance Hyperparameters

When designing experiments, distinguish between:

- **Scientific hyperparameters**: Those you want to scientifically measure the effect of (e.g., model architecture, loss function)
- **Nuisance hyperparameters**: Those that must be tuned for fair comparison but aren't the focus (e.g., learning rate, batch size)

To find optimal values for nuisance hyperparameters, use the hyperparameter search mode:

```bash
python scripts/train.py --config configs/0_baselines/0_simple_baseline.yaml --search_mode --n_trials 20 --lr_range 1e-5,1e-2
```
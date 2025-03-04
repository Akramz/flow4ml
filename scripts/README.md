# Training and Evaluation Scripts

This directory contains scripts for training, evaluating, and analyzing models. These scripts form the core of the Flow4ML workflow and support the iterative machine learning development process.

## Main Scripts

- `train.py`: Train models using configuration files
- `evaluate.py`: Evaluate trained models on test data
- `analyze.py`: Perform error analysis to guide improvements

## Training Models

The `train.py` script is used to train models based on configuration files:

```bash
# Standard training
python train.py --config configs/0_baselines/0_simple_baseline.yaml

# With overrides
python train.py --config configs/0_baselines/0_simple_baseline.yaml --learning_rate 0.002 --batch_size 64

# Hyperparameter search
python train.py --config configs/0_baselines/0_simple_baseline.yaml --search_mode --n_trials 20
```

### Key Training Options

- `--config`: Path to YAML configuration file (required)
- `--task_type`: Task type (base, segmentation, classification, regression)
- `--model_name`: Override model name from config
- `--backbone_name`: Override backbone name
- `--learning_rate`: Override learning rate
- `--batch_size`: Override batch size
- `--max_epochs`: Override maximum epochs
- `--gpu_ids`: Specify GPU IDs to use

### Hyperparameter Search

Use the `--search_mode` flag to enable Optuna-based hyperparameter optimization:

- `--n_trials`: Number of trials to run (default: 20)
- `--search_epochs`: Epochs per trial (default: 10)
- `--lr_range`: Learning rate range, e.g., "1e-5,1e-2"
- `--wd_range`: Weight decay range, e.g., "1e-6,1e-3"
- `--batch_size_range`: Comma-separated batch sizes, e.g., "16,32,64"
- `--disable_tuning`: Parameters to exclude from tuning
- `--frac`: Data fraction to use for faster search

## Evaluating Models

The `evaluate.py` script runs inference and computes metrics:

```bash
python evaluate.py --model_path model_runs/experiment/best.ckpt --test_data path/to/test/data
```

### Key Evaluation Options

- `--model_path`: Path to model checkpoint (required)
- `--test_data`: Path to test data (required)
- `--config`: Path to original config file (optional)
- `--output_dir`: Directory to save results (default: "evaluation_results")
- `--batch_size`: Batch size for evaluation
- `--save_predictions`: Save model predictions to disk
- `--task_type`: Task type (base, segmentation, classification, regression)
- `--gpu_id`: GPU ID to use for evaluation

## Error Analysis

The `analyze.py` script helps identify and visualize model errors:

```bash
python analyze.py --model_path model_runs/experiment/best.ckpt --test_data path/to/test/data
```

### Key Analysis Options

- `--model_path`: Path to model checkpoint (required)
- `--test_data`: Path to test data (required)
- `--output_dir`: Directory to save analysis (default: "analysis_results")
- `--n_worst_samples`: Number of worst samples to visualize (default: 20)
- `--task_type`: Task type (base, segmentation, classification, regression)
- `--gpu_id`: GPU ID to use for analysis

## Extending Scripts

When developing new functionality:

1. Maintain the command-line interface pattern
2. Add descriptive help text for all parameters
3. Keep core functionality modular
4. Add new parameters with sensible defaults

## Custom Scripts

For project-specific needs, create additional scripts in this directory. Common extensions include:

- Data preprocessing scripts
- Custom inference scripts
- Ensemble generation scripts
- Error analysis-specific tools

Follow the same command-line interface pattern and documentation standards used in the core scripts.
# Flow4ML: Earth Water Surface Detection

This project demonstrates using Flow4ML to detect water surfaces from satellite imagery using TorchGeo. It showcases how Flow4ML can be applied to real-world remote sensing tasks.

*Inspired by [Mauricio Cordeiro](https://github.com/cordmaur)*

## Project Overview

This use case focuses on identifying water surfaces from multispectral satellite imagery. The implementation:

- Leverages **TorchGeo** for geospatial data handling and model architectures
- Provides multiple model architectures including UNet and UPerNet with various backbones 
- Utilizes spectral indices (NDWI, NDVI) to enhance water detection capabilities
- Implements automatic dataset download and preprocessing
- Supports visualization for multi-channel satellite imagery

## Setup

### Environment Setup

Create a new environment using conda:

```bash
mamba create -n watermap python=3.13
conda activate watermap
pip install -r requirements.txt
pip install -e .
```

## Running Experiments

### Available Models

We provide several model configurations for water surface detection:

1. **Random Baseline**: A simple random prediction baseline
2. **UNet with ResNext101 Backbone**: A powerful segmentation architecture
3. **UPerNet with ResNet34 Backbone**: An alternative architecture designed for semantic segmentation

### Training Models

To train a model, use the following commands:

```bash
# Random baseline model
python scripts/train.py --config configs/0_baselines/1_random_water_surface.yaml

# UNet with ResNext101 backbone
python scripts/train.py --config configs/0_baselines/2_unet_resnext101_water_surface.yaml

# UPerNet with ResNet34 backbone  
python scripts/train.py --config configs/0_baselines/3_upernet_resnet34_water_surface.yaml
```

### Evaluating Models

To evaluate a trained model:

```bash
python scripts/evaluate.py --model-path model_runs/experiment_name/best.ckpt
```

### Analyzing Results

For detailed error analysis:

```bash
python scripts/analyze.py --model-path model_runs/experiment_name/best.ckpt
```

## Dataset

The Earth Surface Water dataset is automatically downloaded and preprocessed when running the training script for the first time. The dataset consists of multispectral satellite imagery with corresponding water surface masks.

## Model Performance

| Model | IoU | Accuracy | F1 Score |
|-------|-----|----------|----------|
| Random Baseline | ~0.30 | ~0.50 | ~0.45 |
| UNet (ResNext101) | ~0.75 | ~0.90 | ~0.85 |
| UPerNet (ResNet34) | ~0.70 | ~0.88 | ~0.82 |

*Note: Actual performance may vary based on training conditions.*

## Project Structure

```
├── flow/                # Core implementation modules
│   ├── config.py        # Configuration validation
│   ├── datasets.py      # Earth water surface dataset implementation
│   ├── datamodules.py   # PyTorch Lightning data modules
│   ├── models.py        # Model architectures including UNet and UPerNet
│   └── trainers.py      # Training logic and TorchGeo integration
├── configs/             # YAML configuration files
│   └── 0_baselines/     # Baseline water surface detection models
├── scripts/             # Training and evaluation scripts
│   ├── train.py         # Main training script
│   ├── evaluate.py      # Evaluation script
│   └── analyze.py       # Error analysis tools
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
"""
Configuration validator for ML model training pipeline.

Core features:
- Validates model config: Architecture, hyperparameters
- Training parameters: Optimizer, scheduler, loss functions
- Experiment tracking: Auto-generates names, manages output dirs
- Data config: Input formats, batch sizes, augmentations
- Hardware: Multi-GPU support with validation

Example:
   config = TrainerConfig(
       model_name="unet",
       backbone_name="resnet18", 
       input_dirs="data/train.csv",
       in_channels=5,
       batch_size=32
   )
"""

from pathlib import Path
from enum import Enum
from typing import List, Optional, Union, Dict, Any

from pydantic import BaseModel, field_validator, model_validator
import torch

# Get available GPUs for validation
GPUS_AVAILABLE = torch.cuda.device_count()


##################################################
### Define user choices to validate yaml input ###
##################################################


class ModelEnum(str, Enum):
    """Supported model architectures."""

    # Add your model types here
    unet = "unet"
    upernet = "upernet"
    deeplabv3 = "deeplabv3"
    resnet = "resnet"
    transformer = "transformer"
    random = "random"


class ActivationEnum(str, Enum):
    """Supported activation functions."""

    relu = "relu"
    sigmoid = "sigmoid"
    tanh = "tanh"
    leaky_relu = "leaky_relu"
    swish = "swish"
    none = "none"


class WeightEnum(str, Enum):
    """Weight initialization options."""

    pretrained = "pretrained"
    random = "random"
    custom = "custom"


class OptimizerEnum(str, Enum):
    """Supported optimizers."""

    adam = "adam"
    adamw = "adamw"
    sgd = "sgd"
    rmsprop = "rmsprop"


class LossEnum(str, Enum):
    """Supported loss functions."""

    mse = "mse"
    bce = "bce"
    ce = "ce"
    focal = "focal"
    dice = "dice"
    huber = "huber"
    custom = "custom"


class SchedulerEnum(str, Enum):
    """Supported learning rate schedulers."""

    cosine = "cosine"
    plateau = "plateau"
    step = "step"
    linear = "linear"
    none = "none"


class PrecisionEnum(str, Enum):
    """Precision modes for training."""

    float32 = "32-true"
    float16 = "16-mixed"
    bfloat16 = "bf16-mixed"


######################################
### Validate final training config ###
######################################


class TaskTypeEnum(str, Enum):
    """Supported task types."""

    base = "base"
    segmentation = "segmentation"
    classification = "classification"
    regression = "regression"


class TrainerConfig(BaseModel):
    """Validate input from yaml and/or argparse before passing to train.py."""

    # task type
    task_type: TaskTypeEnum = TaskTypeEnum.base

    # model params
    model_name: ModelEnum = ModelEnum.unet
    backbone_name: Optional[str] = None
    weight_init: WeightEnum = WeightEnum.random

    # architecture-specific params (examples)
    num_layers: Optional[int] = None
    hidden_dim: Optional[int] = None
    dropout: float = 0.0

    # optimizer params
    optimizer: OptimizerEnum = OptimizerEnum.adamw
    lr: float = 0.001
    weight_decay: float = 0.01
    patience: int = 10
    scheduler: SchedulerEnum = SchedulerEnum.cosine
    beta1: float = 0.9  # AdamW beta1 parameter
    beta2: float = 0.999  # AdamW beta2 parameter
    t_max: Optional[int] = None  # Cosine annealing parameter

    # loss params
    loss: LossEnum = LossEnum.mse
    loss_weights: Optional[Dict[str, float]] = None  # For multi-task learning

    # data module params
    batch_size: int = 32
    num_workers: int = 4
    validation_split: float = 0.2
    test_split: Optional[float] = None
    augmentations: List[str] = []

    # trainer params
    gpu_ids: List[int] = [0]
    seed: int = 42
    max_epochs: int = 100
    log_dir: str = "logs/"
    output_dir: str = "model_runs/"
    checkpoint_every: int = 10  # Save checkpoint every N epochs
    overwrite: bool = False
    early_stop: bool = True
    early_stop_patience: int = 20

    # input data params
    experiment_name: Optional[str] = None
    experiment_short_name: str = "default_experiment"
    input_dirs: Optional[Union[List[str], str]] = None
    in_channels: Optional[int] = None
    out_channels: Optional[int] = None
    precision: PrecisionEnum = PrecisionEnum.float32

    # Misc parameters
    activation: ActivationEnum = ActivationEnum.relu
    val_ratio: float = 0.1
    resume_from_checkpoint: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"
        use_enum_values = True
        validate_assignment = True

    @field_validator("val_ratio")
    def validate_val_ratio(cls, val_ratio) -> float:
        if not 0 < val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1")
        return val_ratio

    @field_validator("gpu_ids")
    def validate_gpus(cls, gpu_ids) -> List[int]:
        if GPUS_AVAILABLE == 0:
            print("Warning: No GPUs available. Using CPU.")
            return []

        for gpu_id in gpu_ids:
            if gpu_id >= GPUS_AVAILABLE:
                raise ValueError(
                    f"Found only {GPUS_AVAILABLE} GPU(s). Cannot use {gpu_id}."
                )
        print(f"Using the following GPU(s): {gpu_ids}.")
        return gpu_ids

    @field_validator("input_dirs")
    def validate_input_dirs(cls, input_dirs) -> List[str]:
        print("Validating input directories.")
        if isinstance(input_dirs, str):
            input_dirs = [input_dirs]
        for fp in input_dirs:
            fp = Path(fp)
            if not fp.exists():
                raise ValueError(f"{fp} does not exist.")
            if fp.is_file() and not fp.name.endswith((".csv", ".json", ".txt")):
                raise ValueError(f"{fp} is not a valid data file (.csv, .json, .txt).")
        return input_dirs

    @classmethod
    def validate_experiment_name(cls, model):
        if model.experiment_name is not None:
            print("Using provided experiment name.")
            return

        print("Constructing experiment name.")
        short_name = model.experiment_short_name
        model_name = model.model_name
        backbone_name = model.backbone_name if model.backbone_name else "none"
        loss = model.loss
        lr = model.lr
        weight_decay = model.weight_decay
        seed = model.seed
        batch_size = model.batch_size

        experiment_name = (
            f"{short_name}--{model_name}--{backbone_name}--lr_{lr}"
            + f"--wd_{weight_decay}--bs_{batch_size}--loss_{loss}--seed_{seed}"
        )
        model.__dict__["experiment_name"] = experiment_name

    @classmethod
    def validate_output_dir(cls, model):
        print("Validating output directories.")
        log_dir = model.log_dir
        output_dir = model.output_dir
        experiment = model.experiment_name

        output_log_dir = Path(log_dir)
        output_run_dir = Path(output_dir) / experiment

        if (output_log_dir.exists() or output_run_dir.exists()) and not model.overwrite:
            raise ValueError(
                "Output directories already exist and overwrite is not set to true."
            )

        output_log_dir.mkdir(parents=True, exist_ok=True)
        model.__dict__["log_dir"] = str(output_log_dir)

        output_run_dir.mkdir(parents=True, exist_ok=True)
        model.__dict__["output_dir"] = str(output_run_dir)

    @model_validator(mode="after")
    def validate_model(self):
        TrainerConfig.validate_experiment_name(self)
        TrainerConfig.validate_output_dir(self)
        return self

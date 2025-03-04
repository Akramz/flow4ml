# Flow4ML Core Implementation

This directory contains the core implementation of the Flow4ML framework. These modules provide the building blocks for designing, training, and evaluating machine learning models.

## Module Structure

- `config.py`: Configuration validation and management
- `models.py`: Model architecture implementations
- `datasets.py`: Dataset loading and preprocessing
- `datamodules.py`: PyTorch Lightning data modules
- `trainers.py`: Training logic and metrics

## Extending the Framework

### Custom Models

Extend the models by modifying `models.py`:

1. **Add model implementations**: Add new model classes or import from libraries
2. **Update the model factory**: Modify `get_model()` to support new models
3. **Add model-specific parameters**: Update the config validation in `config.py`

Example for adding a new model:
```python
# In models.py
class MyCustomModel(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        # Model implementation
        
    def forward(self, x):
        # Forward pass
        return x
        
# Update the model factory
def get_model(config):
    # ... existing code ...
    elif config.model_name == "my_custom_model":
        return MyCustomModel(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            # Other parameters
        )
```

### Custom Datasets

Extend the data loading by modifying `datasets.py`:

1. **Add dataset classes**: Create new dataset classes for your data
2. **Update transforms**: Add custom preprocessing transforms
3. **Update the dataset factory**: Modify `get_dataset()` to support new datasets

### Custom Tasks

Extend the training logic by modifying `trainers.py`:

1. **Add task classes**: Create new task classes for specific problems
2. **Add custom metrics**: Implement metrics for your specific task
3. **Update the task factory**: Modify `get_task()` to support new tasks

Example for adding a new task:
```python
# In trainers.py
class MyCustomTask(BaseTask):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        # Additional initialization
        
    def training_step(self, batch, batch_idx):
        # Custom training logic
        
    def validation_step(self, batch, batch_idx):
        # Custom validation logic
        
# Update the task factory
def get_task(task_type, model, **kwargs):
    # ... existing code ...
    elif task_type.lower() == "my_custom_task":
        return MyCustomTask(model, **kwargs)
```

## Best Practices

1. **Keep modifications modular**: Isolate changes to specific components
2. **Maintain interfaces**: Preserve function signatures for compatibility
3. **Add tests**: Test new functionality to ensure it works as expected
4. **Document changes**: Add docstrings and update READMEs with new features
5. **Update configurations**: Add example configuration files for new features

## Integration with Scripts

When adding new functionality, make sure to update the relevant scripts in the `scripts/` directory to expose your new features through the command-line interface.

For example, if you add a new task type, update the `train.py` script to recognize and correctly handle the new task type.
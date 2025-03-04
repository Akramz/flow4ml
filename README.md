# Flow4ML: A Framework for Iterative ML Model Development

Flow4ML is a template and framework for systematically developing machine learning models that:
1. Quickly iterates to near-optimal performance
2. Organizes experiments methodically
3. Prioritizes feedback loops and error analysis

This template provides the structure, code, and guidelines to help ML practitioners follow a robust, iterative approach to model development.

## Framework Overview

### High-level Process
1. **Create an experiment tracking sheet** in [Google Docs](https://sheets.google.com/) with an empty results table
2. **Establish baseline** results for simple methods (e.g., random, mode, median, mean, persistence, etc.)
3. **Iterate through experiments**, documenting each in tracking sheet
4. **Adopt good ideas** and take notes when experiments underperform

### Feedback Loop
1. Start with a specific change in mind and implement it
2. Tune the most important hyperparameters before training
   - **Scientific hyperparameters:** Measure effect on performance
   - **Nuisance hyperparameters:** Must be tuned for fair comparisons
   - **Fixed hyperparameters:** Keep constant for now
3. Train the model using that configuration
4. Conduct inference & evaluation
5. If performing better, analyze errors & find top N% most common failures
6. Consider producing a release based on current iteration
7. Brainstorm ideas to reduce mistakes, prioritize and return to step 1

### Techniques for Fast Experimentation
- **Training:** Subsample data, increase batch size, optimize dataloaders
- **Inference:** Subsample test set
- **Evaluation:** Parallelize and distribute evaluation jobs
- **Analysis:** Focus on model collapses for faster error analysis

### Starting with a Baseline
- Use a proven model architecture
- Choose an adaptive optimizer (Adam, AdamW, NAdam)
- Use the largest batch size possible without OOM errors
- Use a cosine learning rate scheduler
- Plot inputs/outputs/predictions for first N epochs
- Checkpoint and evaluate intermediate models for long training runs

## Template Structure

```
├── flow/                # Core implementation modules
│   ├── config.py        # Configuration validation
│   ├── datasets.py      # Data loading utilities
│   ├── datamodules.py   # PyTorch Lightning data modules
│   ├── models.py        # Model architectures
│   └── trainers.py      # Training logic and metrics
├── configs/             # YAML configuration files by research direction
│   ├── 0_baselines/     # Baseline model configurations
│   ├── 1_architectures/ # Testing different model architectures
│   └── ...              # Other experiment directions
├── scripts/             # Training and evaluation scripts
│   ├── train.py         # Main training script
│   ├── evaluate.py      # Evaluation script
│   └── analyze.py       # Error analysis tools
├── notebooks/           # Jupyter notebooks for analysis
└── docs/                # Documentation and guides
```

## Experiment Tracking

For tracking experiment results and notes, we recommend:

1. **Create a Google Sheet** with the following columns:
   - Experiment ID (corresponding to config file name)
   - Description and hypothesis
   - Key parameter changes
   - Results and metrics
   - Observations and insights
   - Next steps

2. This external tracking approach:
   - Keeps the repository clean
   - Makes collaboration easier
   - Provides a central location for results and insights
   - Can be easily shared and updated

3. **Config files** in the `configs/` directory will serve as a record of the experiments you've run. The ordered directory structure makes it easy to see the progression of your research.

## Setup

1. **Environment Setup**

Create a new environment using conda, mamba, or your preferred environment manager:

```bash
mamba create -n your_project_name python=3.10
conda activate your_project_name
pip install -r requirements.txt
pip install -e .
```

2. **Data Organization**

Prepare your dataset files or URLs following one of these approaches:
- CSV file with pointers to input/output data
- Local file paths organized in directories
- Remote data URLs with access tokens

## Running Experiments

### 1. Train a Model

```bash
python scripts/train.py --config configs/0_baselines/0_simple_baseline.yaml
```

### 2. Evaluate

```bash
python scripts/evaluate.py --model-path model_runs/experiment_name/best.ckpt --test-data path/to/test
```

### 3. Analyze Errors

```bash
python scripts/analyze.py --model-path model_runs/experiment_name/best.ckpt --test-data path/to/test
```

### 4. Hyperparameter Search

```bash
python scripts/train.py --config configs/0_baselines/0_simple_baseline.yaml --search_mode --n_trials 20
```

## Customize for Your Project

1. **Define your data**:
   - Update `datasets.py` with your data loading logic
   - Configure input and output formats

2. **Choose/implement models**:
   - Select from standard models or add custom architectures in `models.py`
   - Configure via YAML files

3. **Set evaluation metrics**:
   - Customize metrics in `trainers.py` for your specific task
   - Add task-specific visualizations

4. **Document your process**:
   - Use your external tracking sheet to record iterations
   - Keep error analysis for each significant improvement

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

MIT License

Copyright (c) 2023 Flow4ML

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

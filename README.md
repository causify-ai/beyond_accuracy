# Beyond Accuracy: A Stability-Aware Metric for Multi-Horizon Forecasting

This repository contains the code for the paper titled "Beyond Accuracy: A Stability-Aware Metric for Multi-Horizon Forecasting" by Chutian Ma, Grigorii Pomazkin, Giacinto Paolo Saggese, and Paul Smith.

## Overview

This project introduces the **Forecast Accuracy and Coherence (AC) Score**, a novel metric for evaluating probabilistic multi-horizon forecasts that accounts for both accuracy and stability. Unlike traditional metrics focused on individual horizons, the AC Score measures how consistently models predict the same future events as forecast origins change along with multi-horizon accuracy.

## Key Results

Our AC-optimized SARI models achieve:
- **91.1% reduction** in forecast volatility for the same target timestamps
- **Up to 26% median improvement** in medium to long horizon accuracy
- Modest one-step-ahead accuracy trade-off (7.5% on average)

## Repository Contents

```
.
├── notebooks/                      # Jupyter notebooks for analysis
├── outcomes/                       # Output directory for results
├── helpers/                        # Helper functions 
├── forecast_metric_utility.py      # Training scripts
├── differentiable_arima.py         # Differentiable SARIMA implementation
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Installation

### Setup

1. Clone this repository:
```bash
git clone https://github.com/causify-ai/beyond_accuracy.git
cd beyond_accuracy
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

The main dependencies include:
- `torch` - PyTorch for differentiable model implementation
- `pmdarima` - Auto-ARIMA for hyperparameter selection
- `statsmodels` - Traditional SARIMA baseline models
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `matplotlib` - Visualization
- `fev` - Interface for benchmark datasets, including M4

Full dependencies are listed in `requirements.txt`.

## Data

The experiments use the M4 Hourly benchmark dataset.

## Usage to Replicate Paper Results

To reproduce the main results from the paper:

```bash
python run_experiment.py --batch_size 3 --num_threads 8 --dst_dir "./outcomes/...(subfolder name)"
```

This will:
1. Load the M4 Hourly dataset and divide into batches
2. Execute the following in parallel
3. Split each series into 60% training / 40% test
4. Use auto-ARIMA to select hyperparameters
5. Train traditional MLE-based SARI models (baseline)
6. Train AC-optimized SARI models
7. Generate out-of-sample forecasts
8. Compute evaluation metrics
9. Save results to the specified dst_dir folder

**Note:** The process takes approximately 12-20 hours on 8 core CPUs. To run partial experiments, additional argument can be added to set the size of experiment. For example, using `--num_test 50` will randomly select 50 time series from the dataset (default seed 42). 

## Implementation Details

### AC Score Metric

The Forecast AC Score combines two components:

1. **Accuracy term:** Multi-horizon energy score with horizon-specific weights
2. **Stability term:** Energy distance between forecasts targeting the same timestamp

The metric is implemented as:
```
AC_score = Accuracy + λ × Stability
```

where λ is the stability multiplier (default: 0.5).

### Differentiable SARIMA

The SARI model is implemented in PyTorch with:
- Autoregressive and seasonal autoregressive coefficients as learnable parameters
- Initialization from auto-ARIMA hyperparameter search

### Horizon Weights

By default, we use linear decay weights:
```
w(h) = 1 - h/H
```
, which emphasizes shorter horizons while maintaining awareness of longer horizons.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ma2026beyond,
  title={Beyond Accuracy: A Stability-Aware Metric for Multi-Horizon Forecasting},
  author={Ma, Chutian and Pomazkin, Grigorii and Saggese, Giacinto Paolo and Smith, Paul},
  journal={arXiv preprint arXiv:2601.10863},
  year={2026}
}
```

## Requirement 
Python version 3.12 or higher.
See requirements.txt for required packages.

## Contact

For questions or issues, please open an issue on GitHub or contact:
- Chutian Ma: c.ma@causify.ai
- Grigorii Pomazkin: g.pomazkin@causify.ai
- Giacinto Paolo Saggese: gp@causify.ai
- Paul Smith: paul@causify.ai

## License

This project is licensed under the Apache License 2.0.
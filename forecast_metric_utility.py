"""
Provide helper functions for computing multi-horizon forecast stability
metrics.

Import as:

import forecast_metric_utility as fmutil
"""

import logging
from typing import Any, Dict, Optional, Tuple
import os
import torch

try:
    from torch.cuda.amp import GradScaler, autocast
except ImportError:
    autocast = None
    GradScaler = None
import random

import numpy as np
import pmdarima
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

import differentiable_arima

_LOG = logging.getLogger(__name__)


def compute_ensembled_energy_score_tensor(
    true_outcomes: torch.Tensor, 
    forecast_ensemble: torch.Tensor, 
    relative_weights: Optional[torch.Tensor] = None,
):
    """
    Compute time averaged energy score of rolling forecasts. ES(F,y) =
    E_F[|X−y|] − 1/2*E_F[|X−X′|] where || is the Euclidean distance (L2)

    :param true_outcomes: true outcome time series
    :param forecast_ensemble: rolling forecasts tensor of shape
        (n_rolling, n_horizon, n_sample)
    :param relative_weights: weights per horizon of shape (n_horizon,)
    :returns: mean_rolling_energy_score as a tensor
    """
    n_rolling, n_horizon, n_sample = forecast_ensemble.shape
    device = forecast_ensemble.device
    # Validate
    if n_rolling + n_horizon - 1 > len(true_outcomes):
        raise ValueError("True outcome is too short to cover all the forecasts.")
    # Set up weights
    if relative_weights is None:
        weights = torch.ones(n_horizon, device=device)
    else:
        weights = relative_weights.to(device)
    w = weights.reshape(1, -1, 1)
    # Set the power of Lp norm in the metric.
    p = 2.0
    # Extract true outcomes
    indices = (
        torch.arange(n_horizon, device=device)[None, :]
        + torch.arange(n_rolling, device=device)[:, None]
    )
    y_true_rolling = true_outcomes[indices]  # (n_rolling, n_horizon)
    # Precision term
    diff_true = forecast_ensemble - y_true_rolling[:, :, None]
    weighted_diff_true = diff_true * w
    precision = torch.linalg.vector_norm(weighted_diff_true, ord=p, dim=1)
    precision_mean = precision.mean(dim=1)
    # Sharpness term - vectorized
    w_sharp = weights.reshape(1, -1, 1, 1)
    y_preds_i = forecast_ensemble[:, :, :, None]
    y_preds_j = forecast_ensemble[:, :, None, :]
    diff_pairs = (y_preds_i - y_preds_j) * w_sharp
    sharpness_matrix = torch.linalg.vector_norm(diff_pairs, ord=p, dim=1)
    # More efficient upper triangle extraction
    # Create mask once
    upper_triangle_mask = torch.triu(
        torch.ones((n_sample, n_sample), dtype=torch.bool, device=device),
        diagonal=1,
    )
    # Vectorized extraction and mean
    # Expand mask to batch dimension
    masked_sharpness = sharpness_matrix[
        :, upper_triangle_mask
    ]  # (n_rolling, n_pairs)
    sharpness_mean = masked_sharpness.mean(dim=1)  # (n_rolling,)
    # Energy scores
    fixed_origin_energy_scores = precision_mean  # - 0.5 * sharpness_mean
    mean_rolling_energy_score = fixed_origin_energy_scores.mean()
    return mean_rolling_energy_score


def compute_ensembled_energy_distance_tensor(
    forecast_ensemble: torch.Tensor, relative_weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute energy distance to use as a loss function for PyTorch.

    :param forecast_ensemble: rolling forecasts tensor of shape
        (n_rolling, n_horizon, n_sample)
    :param relative_weights: weight per horizon of shape (n_horizon-1, )
    :return: stability score of the rolling forecasts (as a tensor)
    """
    n_rolling, n_horizon, n_sample = forecast_ensemble.shape
    device = forecast_ensemble.device
    if n_rolling < 2:
        raise ValueError(
            "Need at least 2 rolling origins to compute energy distance"
        )
    # Set up weights
    if relative_weights is None:
        weights = torch.ones(n_horizon - 1, device=device)
    else:
        weights = relative_weights.to(device)
    w = weights.reshape(1, -1, 1)
    p = 2.0
    # Extract aligned forecast pairs
    y_given_t = forecast_ensemble[
        :-1, 1:, :
    ]  # (n_rolling-1, n_horizon-1, n_sample)
    y_given_t_plus_1 = forecast_ensemble[
        1:, :-1, :
    ]  # (n_rolling-1, n_horizon-1, n_sample)
    # Closeness term
    diff_closeness = (y_given_t - y_given_t_plus_1) * w
    closeness = torch.linalg.vector_norm(diff_closeness, ord=p, dim=1)
    closeness_mean = closeness.mean(dim=1)
    # Sharpness terms
    w_sharp = weights.reshape(1, -1, 1, 1)
    # For y_given_t
    y_t_i = y_given_t[:, :, :, None]
    y_t_j = y_given_t[:, :, None, :]
    diff_t = (y_t_i - y_t_j) * w_sharp
    sharpness_t_matrix = torch.linalg.vector_norm(diff_t, ord=p, dim=1)
    # For y_given_t_plus_1
    y_t1_i = y_given_t_plus_1[:, :, :, None]
    y_t1_j = y_given_t_plus_1[:, :, None, :]
    diff_t1 = (y_t1_i - y_t1_j) * w_sharp
    sharpness_t1_matrix = torch.linalg.vector_norm(diff_t1, ord=p, dim=1)
    # Combined sharpness
    total_sharpness_matrix = 0.5 * (sharpness_t_matrix + sharpness_t1_matrix)
    # Vectorized upper triangle extraction
    upper_triangle_mask = torch.triu(
        torch.ones((n_sample, n_sample), dtype=torch.bool, device=device),
        diagonal=1,
    )
    masked_sharpness = total_sharpness_matrix[:, upper_triangle_mask]
    sharpness_mean = masked_sharpness.mean(dim=1)
    # Energy distance
    fixed_origin_energy_distance = closeness_mean  # - sharpness_mean
    ensemble_energy_distance = fixed_origin_energy_distance.mean()
    return ensemble_energy_distance


def custom_metric(
    forecast_ensemble: torch.Tensor,
    y_true: torch.Tensor,
    model: differentiable_arima.DifferentiableARIMA,
    relative_weights_acc: torch.Tensor,
    relative_weights_stb: torch.Tensor,
    mult_stability: float,
):
    """
    Custom metric using the weighted sum of the accuracy and stability scores.

    :param forecast_ensemble: (n_forecasts, nsimulations, repetitions)
        torch.Tensor
    :param y_true: Full time series torch.Tensor
    :param model: ARIMA model (to get starting index)
    :param relative_weights_acc: Relative weights for accuracy
    :param relative_weights_stb: Relative weights for stability
    :param mult_stability: Multiplier for stability term
    :return: loss value
    """
    # Calculate minimum history
    min_history = model.min_history
    # Energy score expects true_outcomes to start from min_history + 1
    # because forecast_ensemble[0] predicts y[min_history+1:min_history+1+nsimulations]
    true_outcomes = y_true[min_history:]  # Start from first target
    # Compute energy score (lower is better)
    accuracy = compute_ensembled_energy_score_tensor(
        true_outcomes, forecast_ensemble, relative_weights_acc
    )
    stability = compute_ensembled_energy_distance_tensor(
        forecast_ensemble, relative_weights_stb
    )
    metric = accuracy + mult_stability * stability
    return metric


def _uniform_weight(length: int):
    """
    Generate uniform weights of 1 over the specified length.

    :param length: length of the weight vector
    :return: torch.Tensor of shape (length,)
    """
    weights = torch.ones(length, dtype=torch.float32)
    return weights


def _linear_weight(length: int):
    """
    Generate linear weights from 1 to 0 over the specified length.

    :param length: length of the weight vector
    :return: torch.Tensor of shape (length,)
    """
    weights = torch.tensor(np.linspace(1, 0, length), dtype=torch.float32)
    return weights


def _exponential_weight(length: int, decay_rate: float = 5.0):
    """
    Generate decaying exponential weights over the specified length.

    :param length: length of the weight vector
    :param decay_rate: rate of exponential decay
    :return: torch.Tensor of shape (length,)
    """
    weights = torch.tensor(np.exp(np.linspace(0, -decay_rate, length)), dtype=torch.float32)
    return weights


def _hyperbolic_weight(length: int, decay_rate: float = 1.0):
    """
    Generate hyperbolic weights over the specified length.

    :param length: length of the weight vector
    :param decay_rate: rate of hyperbolic decay
    :return: torch.Tensor of shape (length,)
    """
    weights = torch.tensor([1 / (i * decay_rate + 1) for i in range(length)], dtype=torch.float32)
    return weights

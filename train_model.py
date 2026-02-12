"""
Provide code for training differentiable ARIMA models using MLE and custom metrics.

Import as:

import train_model
"""

import logging
from typing import Any, Dict, Optional, Tuple
import torch

try:
    from torch.cuda.amp import GradScaler, autocast
except ImportError:
    autocast = None
    GradScaler = None
import random

import numpy as np
import pmdarima
from statsmodels.tsa.arima.model import ARIMA

import differentiable_arima
import forecast_metric_utility as fmutil

_LOG = logging.getLogger(__name__)


def train_arima_optimized(
    model,
    y_train: torch.Tensor,
    model_config: Dict[str, Any],
    nsimulations: int,
    last_forecast_origin: Optional[int] = None,
    repetitions: int = 100,
    lr: float = 0.01,
    relative_weights_acc: Optional[torch.Tensor] = None,
    relative_weights_stb: Optional[torch.Tensor] = None,
    mult_stability: float = 1.0,
    device: str = "cuda",
    batch_size: int = 32,
    use_amp: bool = True,
    convergence_window: int = 100,
    convergence_threshold: float = 1e-4,
    max_epochs: int = 3000,
    min_epochs: int = 200,
) -> Tuple[differentiable_arima.DifferentiableARIMA, Dict[str, Any]]:
    """
    Train the differentiable ARIMA model with automatic convergence detection.

    :param model: DifferentiableARIMA instance
    :param y_train: training time series data
    :param model_config: model hyperparameters
    :param nsimulations: number of steps to forecast
    :param last_forecast_origin: index of the last forecast origin in y_train;
        if None, default to len(y_train) - nsimulations - 1
    :param repetitions: number of sample paths per forecast origin
    :param epochs: initial number of training epochs
    :param lr: learning rate for optimizer
    :param relative_weights_acc: relative weights for accuracy metric
    :param relative_weights_stb: relative weights for stability metric
    :param mult_stability: multiplier for stability metric
    :param device: device to use ('cuda' or 'cpu')
    :param batch_size: batch size for rolling forecast computation
    :param use_amp: whether to use automatic mixed precision
        DO NOT set to True if using CPU!
    :param convergence_window: number of epochs to check for convergence
    :param convergence_threshold: relative change threshold for convergence
    :param max_epochs: maximum number of epochs
    :param min_epochs: minimum epochs before checking convergence
    :return: trained model and training history
    """
    # Load training settings.
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    p, d, q = model_config["p"], model_config["d"], model_config["q"]
    P, D, Q, m = (
        model_config["P"],
        model_config["D"],
        model_config["Q"],
        model_config["m"],
    )
    y_train_torch = torch.tensor(y_train, dtype=torch.float32).to(device)
    # Get weights if none provided.
    if relative_weights_acc is None or relative_weights_stb is None:
        relative_weights_acc, relative_weights_stb = _get_weights("linear", nsimulations)
    # Calculate minimum history using expanded polynomial in eval mode.
    model.eval()
    with torch.no_grad():
        ar_coefs = model._get_expanded_ar_coefs()
        min_history = len(ar_coefs) if len(ar_coefs) > 0 else 1
    # Back to train mode.
    model.train()
    if not last_forecast_origin:
        last_forecast_origin = len(y_train) - nsimulations - 1
    # Make sure we have enough data.
    if len(y_train) < min_history:
        _LOG.warning(
            f"Error: Need at least {min_history} data points to start forecasting, have {len(y_train)}."
        )
        return
    min_required = last_forecast_origin + nsimulations
    if len(y_train) < min_required:
        _LOG.warning(
            f"Need at least {min_required} data points to compare with forecasts, have {len(y_train)}."
        )
        last_forecast_origin = max(0, len(y_train) - min_history - nsimulations)
        _LOG.info(f"Adjusted last_forecast_origin to {last_forecast_origin}")
    # total_params = sum(param.numel() for param in model.parameters())
    # Initiate optimizer (exclude log_sigma from optimization).
    trainable_params = [
        p for name, p in model.named_parameters() if "log_sigma" not in name
    ]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )
    scaler = GradScaler() if use_amp else None
    # Initiate training tracking.
    best_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    training_history = {
        "loss_history": [],
        "final_params": None,
        "converged": False,
        "convergence_epoch": None,
        "config": {
            "p": p,
            "d": d,
            "q": q,
            "P": P,
            "D": D,
            "Q": Q,
            "m": m,
            "lr": lr,
            "initial_epochs": min_epochs,
            "nsimulations": nsimulations,
            "repetitions": repetitions,
        },
    }

    def check_convergence(loss_history, window, threshold):
        """
        Check if training has converged based on recent loss changes.

        :param loss_history: historical loss values
        :param window: number of recent epochs to consider
        :param threshold: relative change threshold
        :return: whether the training has converged
        """
        if len(loss_history) < window:
            return False
        recent_losses = loss_history[-window:]
        mean_loss = np.mean(recent_losses)
        # Check relative change.
        if mean_loss == 0:
            return False
        # Also check if loss is still decreasing.
        first_half = np.mean(recent_losses[: window // 2])
        second_half = np.mean(recent_losses[window // 2 :])
        relative_improvement = (first_half - second_half) / abs(first_half)
        # Converged if improvement is minimal.
        return relative_improvement < threshold

    epoch = 0
    while epoch < max_epochs:
        optimizer.zero_grad()
        # Compute residuals once per epoch (for MA components).
        # TODO(Chutian): probably can be optimized to use the one-step-ahead forecast
        # of last epoch. Although in that case the residuals will be computed using
        # slightly outdated parameters.
        if q > 0 or Q > 0:
            # Disable gradient tracking.
            with torch.no_grad():
                residuals_cached = model.compute_residuals(y_train_torch)
        else:
            residuals_cached = None
        # Forward pass.
        if use_amp:
            with autocast():
                forecast_ensemble = model.rolling_forecast_batched(
                    y_train_torch,
                    nsimulations,
                    repetitions,
                    last_forecast_origin=last_forecast_origin,
                    batch_size=batch_size,
                    residuals_cached=residuals_cached,
                )
                if forecast_ensemble.shape[0] == 0:
                    _LOG.warning(
                        "No forecasts generated! Check data length and parameters."
                    )
                    break
                loss = fmutil.custom_metric(
                    forecast_ensemble,
                    y_train_torch,
                    model,
                    relative_weights_acc=relative_weights_acc,
                    relative_weights_stb=relative_weights_stb,
                    mult_stability=mult_stability,
                )
            if not torch.isnan(loss):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # TODO(Chutian): Try normalizing const so that it has similar scale as other coefs.
                scaler.step(optimizer)
                scaler.update()
            else:
                _LOG.warning(
                    f"Warning: NaN loss at epoch {epoch+1}, skipping update"
                )
                epoch += 1
                continue
        else:
            forecast_ensemble = model.rolling_forecast_batched(
                y_train_torch,
                nsimulations,
                repetitions,
                last_forecast_origin=last_forecast_origin,
                batch_size=batch_size,
                residuals_cached=residuals_cached,
            )
            if forecast_ensemble.shape[0] == 0:
                _LOG.warning(
                    "No forecasts generated! Check data length and parameters."
                )
                break
            loss = fmutil.custom_metric(
                forecast_ensemble,
                y_train_torch,
                model,
                relative_weights_acc=relative_weights_acc,
                relative_weights_stb=relative_weights_stb,
                mult_stability=mult_stability,
            )
            if not torch.isnan(loss):
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
            else:
                _LOG.warning(
                    f"Warning: NaN loss at epoch {epoch+1}, skipping update"
                )
                epoch += 1
                continue
        scheduler.step(loss)
        training_history["loss_history"].append(loss.item())
        # Track best model.
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
        # Check for learning rate changes.
        current_lr = optimizer.param_groups[0]["lr"]
        # Detailed output every 100 epochs.
        if (epoch + 1) % 100 == 0:
            _LOG.info(f"Epoch {epoch+1}/{max_epochs}")
            _LOG.info(f"  Loss: {loss.item():.6f} (Best: {best_loss:.6f})")
            _LOG.info(f"  LR: {current_lr:.6e}")
        # Check convergence after minimum epochs.
        if epoch >= min_epochs:
            if check_convergence(
                training_history["loss_history"],
                convergence_window,
                convergence_threshold,
            ):
                training_history["converged"] = True
                training_history["convergence_epoch"] = epoch + 1
                break
        # Next epoch.
        epoch += 1
    # Load best model.
    if best_model_state is not None:
        model.load_state_dict(
            {k: v.to(device) for k, v in best_model_state.items()}
        )
        _LOG.info("Loaded best model from training")
    # Record final state.
    training_history["final_params"] = {
        "ar_params": (
            model.ar_params.data.cpu().numpy().tolist() if p > 0 else None
        ),
        "ma_params": (
            model.ma_params.data.cpu().numpy().tolist() if q > 0 else None
        ),
        "seasonal_ar_params": (
            model.seasonal_ar_params.data.cpu().numpy().tolist()
            if P > 0
            else None
        ),
        "seasonal_ma_params": (
            model.seasonal_ma_params.data.cpu().numpy().tolist()
            if Q > 0
            else None
        ),
        "const": model.const.item() if model.with_intercept else 0.0,
        "sigma": model.sigma.item(),
    }
    _LOG.info("\nStationarity Check:")
    model.eval()
    with torch.no_grad():
        # Get expanded AR coefficients (without differencing terms).
        ar_coefs_expanded = model._get_expanded_ar_coefs(
            include_differencing=False
        )
        if len(ar_coefs_expanded) > 0:
            # Build polynomial coefficients for np.roots.
            # We need the coefficients of the polynomial 1 - phi_1 * z - phi_2 * z^2 - ...
            # from the highest order to lowest.
            # So we negate ar_coefs_expanded, reverse its order and add a constant one at the end.
            ar_poly_coeffs = np.concatenate(
                [-ar_coefs_expanded.flip(0).cpu().numpy(), [1]]
            )
            roots = np.roots(ar_poly_coeffs)
            root_mags = np.abs(roots)
            _LOG.info(
                f"  Expanded AR polynomial has {len(ar_coefs_expanded)} coefficients"
            )
            _LOG.info(
                f"\n  Checking polynomial: 1 - phi_1·z - ... - phi_{len(ar_coefs_expanded)}·z^{len(ar_coefs_expanded)}"
            )
            _LOG.info(
                f"  For np.roots: [{ar_poly_coeffs[0]:.4f}, ..., {ar_poly_coeffs[-1]:.4f}]"
            )
            _LOG.info("\n  AR polynomial roots (5 closest to unit circle):")
            sorted_indices = np.argsort(root_mags)
            for i in sorted_indices[:5]:
                root = roots[i]
                mag = root_mags[i]
                _LOG.info(f"    z = {root:.4f}, |z|={mag:.4f}")
            if len(roots) > 5:
                _LOG.info(f"    ... and {len(roots)-5} more roots")
            min_mag = root_mags.min()
            if np.all(root_mags > 1):
                _LOG.info(f"Model is STATIONARY (min |z|={min_mag:.4f})")
            else:
                _LOG.info(f"Model is NON-STATIONARY (min |z|={min_mag:.4f})")
    model.train()
    return model, training_history


def rolling_forecast_with_statsmodel_arima(
    fitted_result, new_ts, forecast_horizon=24
):
    """
    Use a fitted ARIMA model to perform rolling multi-step forecasts on a new
    time series.

    :param fitted_result: a fitted ARIMA result object (from
        model.fit()) :parma new_ts: new time series to forecast on
        (different from training data)
    :param forecast_horizon: number of steps ahead to forecast at each
        origin
    :return: rolling forecasts of shape (n_forecasts, forecast_horizon)
    """
    n = len(new_ts)
    # Determine minimum history needed using expanded polynomial approach
    p = fitted_result.model.order[0]  # AR order
    d = fitted_result.model.order[1]  # Differencing order
    P = fitted_result.model.seasonal_order[0]  # Seasonal AR order
    D = fitted_result.model.seasonal_order[1]  # Seasonal differencing order
    m = fitted_result.model.seasonal_order[3]  # Seasonal period
    # Calculate min_history from expanded polynomial.
    min_history = calculate_min_history_statsmodel(p, d, P, D, m)
    # Set forecast origins.
    first_forecast_origin = min_history - 1
    last_forecast_origin = n - forecast_horizon - 1
    if first_forecast_origin > last_forecast_origin:
        raise ValueError(
            f"Time series too short. Need at least {min_history + forecast_horizon} points, "
            f"got {n}. (min_history={min_history}, horizon={forecast_horizon})"
        )
    forecasts = []
    # Rolling forecasts iterating over forecast origins.
    for forecast_origin in range(first_forecast_origin, last_forecast_origin + 1):
        # Use data up to and including forecast_origin.
        history = new_ts[: forecast_origin + 1]
        # Apply the fitted parameters to the history.
        result_applied = fitted_result.apply(history)
        # Forecast the next forecast_horizon steps.
        forecast = result_applied.forecast(steps=forecast_horizon)
        forecasts.append(forecast)
    # Convert to array.
    forecasts = np.array(forecasts)
    _LOG.info(f"Completed {len(forecasts)} rolling forecasts with Statsmodel.")
    return forecasts


def calculate_min_history_statsmodel(p, d, P, D, m):
    """
    Calculate minimum history needed based on expanded AR polynomial.

        The expanded polynomial (1-L)^d (1-L^m)^D (1-phi_1*L-...-phi_p*L^p)(1-Phi_1*L^m-...-Phi_P*L^(P*m))
        determines the maximum lag needed.

    :param p: auto-regressive order
    :param d: differencing order
    :param P: seasonal auto-regressive order
    :param D: seasonal differencing order
    :param m: period of season
    :return: minimum history equal to the highest order of the expanded polynomial
    """
    # Start with degree 0
    max_degree = 0
    # (1-L)^d contributes degree d
    max_degree += d
    # (1-L^m)^D contributes degree D*m
    max_degree += D * m
    # (1-phi_1*L-...-phi_p*L^p) contributes degree p
    max_degree += p
    # (1-Phi_1*L^m-...-Phi_P*L^(P*m)) contributes degree P*m
    max_degree += P * m
    # Minimum history is the maximum lag + 1 (to have at least one observation at that lag)
    min_history = max(max_degree, 1)
    return min_history


def train_ARIMA_using_two_objectives(
    hyper_params: Dict[str, Any],
    train_ts: np.ndarray,
    validate_ts: np.ndarray,
    nsimulations: int,
    relative_weights_acc: torch.Tensor,
    relative_weights_stb: torch.Tensor,
    mult_stability,
    *,
    nrepetitions: int = 2,
    min_epochs: int = 200,
    max_epochs: int = 3000,
    use_amp: bool = False,
    device_type: Optional[str] = "cpu",
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
]:
    """
    Wrapper for training seasonal integrated AR model using statsmodel's
    default fit() method and using our custom metric as objective functions.

    :param hyper_params: hyper-parameters for the ARIMA model
    :param train_ts: training data
    :param validate_ts: validation data
    :param nsimulations: horizon to forecast
    :param nrepetitions: number of sample path to draw Note that we
        don't sample noise so sample size is currently pointless. But
        the output still needs to have the sample dimension for
        compatibility with metric computation functions. Do not change
        this value.
    :param min_epochs: minimum epochs to run gradient descent
    :param max_epochs: maximum epochs to run gradient descent
    :param use_amp: whether to use automatic mixed precision Do NOT set
        to True if using cpu!
    :param device: device type to run the optimization
    :return: forecast accuracy score of (1) statsmodels' arima on
        training set (2) statsmodels' arima on validation set (3)
        metric-optimized arima on training set (4) metric-optimized
        arima on validation set (5) out-of-sample rolling forecasts of
        statsmodels' arima (6) out-of-sample rolling forecasts of
        metric-optimized arima
    """
    p, d, q = hyper_params["p"], hyper_params["d"], hyper_params["q"]
    P, D, Q, m = (
        hyper_params["P"],
        hyper_params["D"],
        hyper_params["Q"],
        hyper_params["m"],
    )
    # (1) Fit default ARIMA model.
    arima_with_noise = ARIMA(
        train_ts, order=(p, d, q), seasonal_order=(P, D, Q, m)
    )
    fitted_arima = arima_with_noise.fit()
    # (1.1) In-sample test.
    rolling_forecast_results = rolling_forecast_with_statsmodel_arima(
        fitted_arima, train_ts, forecast_horizon=nsimulations
    )
    # Convert to tensor and create a sample dimension.
    forecast_ensemble_default = (
        torch.tensor(rolling_forecast_results).unsqueeze(-1).repeat(1, 1, 2)
    )
    # Get actual time series.
    min_history = calculate_min_history_statsmodel(p, d, P, D, m)
    observed_train_ts = train_ts[min_history:]
    # Compute custom metrics.
    accuracy = fmutil.compute_ensembled_energy_score_tensor(
        observed_train_ts,
        forecast_ensemble_default,
        relative_weights_acc,
    )
    stability = fmutil.compute_ensembled_energy_distance_tensor(
        forecast_ensemble_default,
        relative_weights_stb,
    )
    metric = accuracy + mult_stability * stability
    # Store loss and parameters.
    train_result_default_metric = {
        "loss": metric,
        "accuracy_score": accuracy,
        "stability_score": stability,
        "coefficients": fitted_arima.params,  # (intercept, ar_coef, s_ar_coef, noise)
    }
    # (1.2) Out-of-sample test.
    rolling_forecast_results = rolling_forecast_with_statsmodel_arima(
        fitted_arima, validate_ts, forecast_horizon=nsimulations
    )
    # Convert to tensor and create a sample dimension with two duplicated point forecasts.
    forecast_ensemble_default = (
        torch.tensor(rolling_forecast_results).unsqueeze(-1).repeat(1, 1, 2)
    )
    # Get actual time series.
    min_history = calculate_min_history_statsmodel(p, d, P, D, m)
    first_possible_target = min_history
    observed_validate_ts = validate_ts[min_history:]
    # Compute custom metrics.
    accuracy = fmutil.compute_ensembled_energy_score_tensor(
        observed_validate_ts,
        forecast_ensemble_default,
        relative_weights_acc,
    )
    stability = fmutil.compute_ensembled_energy_distance_tensor(
        forecast_ensemble_default,
        relative_weights_stb,
    )
    metric = accuracy + mult_stability * stability
    # Store loss and parameters.
    validate_result_default_metric = {
        "loss": metric,
        "accuracy_score": accuracy,
        "stability_score": stability,
        "coefficients": fitted_arima.params,  # (intercept, ar_coef, s_ar_coef, noise)
        "true_outcome": observed_validate_ts,
    }
    # (2) Fit custom metric ARIMA model.
    device = torch.device(
        device_type
        if device_type
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    with_intercept = False if d > 0 or D > 0 else True
    differentiable_model = differentiable_arima.DifferentiableARIMA(
        p=p, d=d, q=q, P=P, D=D, Q=Q, m=m, with_intercept=with_intercept
    ).to(device)
    trained_model, training_history = train_arima_optimized(
        differentiable_model,
        train_ts,
        hyper_params,
        nsimulations=nsimulations,
        repetitions=nrepetitions,
        relative_weights_acc=relative_weights_acc,
        relative_weights_stb=relative_weights_stb,
        mult_stability=mult_stability,
        lr=0.1,
        device=device,
        batch_size=64,
        use_amp=use_amp,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        convergence_threshold=1e-4,
    )
    train_result_custom_metric = training_history
    # (2.1) Perform out-of-sample test.
    validate_tensor = torch.tensor(validate_ts).to(device=device)
    forecast_ensemble_custom = trained_model.rolling_forecast_batched(
        validate_tensor, nsimulations, nrepetitions, batch_size=64
    )
    validate_ts_truncated = validate_ts[first_possible_target:]
    # Convert to tensor.
    validate_tensor_truncated = torch.tensor(validate_ts_truncated).to(
        device=device
    )
    # Compute OOS metrics.
    accuracy = fmutil.compute_ensembled_energy_score_tensor(
        validate_tensor_truncated,
        forecast_ensemble_custom,
        relative_weights_acc,
    )
    stability = fmutil.compute_ensembled_energy_distance_tensor(
        forecast_ensemble_custom,
        relative_weights_stb,
    )
    metric = accuracy + mult_stability * stability
    validate_result_custom_metric = {
        "loss": metric.detach().cpu(),
        "accuracy_score": accuracy.detach().cpu(),
        "stability_score": stability.detach().cpu(),
        "true_outcome": validate_ts_truncated,
    }
    return (
        train_result_default_metric,
        validate_result_default_metric,
        train_result_custom_metric,
        validate_result_custom_metric,
        forecast_ensemble_default,
        forecast_ensemble_custom,
    )


def set_torch_seed(seed=42):
    """
    Set random seeds for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Make CUDA operations deterministic.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_weights(weight_type: str, nsimulations: int) -> torch.Tensor:
    """
    Get relative weights tensor based on specified type.

    :param weight_type: type of weights ('uniform', 'linear', 'exponential')
    :param nsimulations: number of simulation steps
    :return: weights tensor of shape (nsimulations,)
    """
    if weight_type == "uniform":
        weight_acc = fmutil._uniform_weight(nsimulations)
        weight_stb = fmutil._uniform_weight(nsimulations-1)
    elif weight_type == "linear":
        weight_acc = fmutil._linear_weight(nsimulations)
        weight_stb = fmutil._linear_weight(nsimulations-1)
    elif weight_type == "exponential":
        weight_acc = fmutil._exponential_weight(nsimulations)
        weight_stb = fmutil._exponential_weight(nsimulations-1)
    elif weight_type == "hyperbolic":
        weight_acc = fmutil._hyperbolic_weight(nsimulations)
        weight_stb = fmutil._hyperbolic_weight(nsimulations-1)
    else:
        raise ValueError(f"Undefined weight type: {weight_type}")
    return weight_acc, weight_stb


def run_on_dataset(
    data,  # type: datasets.Dataset
    selected_ids: np.ndarray,
    train_ratio: float,
    *,
    weight_type: str = "linear",
    nsimulations: int = 24,
    nrepetitions: int = 2,
    mult_stability: float = 0.5,
    seed: int = 42,
    **kwargs
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Run 'train_ARIMA_using_two_objectives()' on a subset of the provided
    dataset.

    :param data: dataset containing all time series
    :param selected_ids: list containing the ids of time series for
        experiment
    :param train_ratio: ratio of train-test-split
    :param weight_type: type of the horizon discounting weights,
        from ("uniform", "linear", "exponential", "hyperbolic")
    :param nsimulations: same as in train_arima_using_two_objectives()
    :param nrepetitions: same as in train_arima_using_two_objectives()
    :param mult_stability: same as in train_arima_using_two_objectives()
    :param seed: random seed for all pytorch and numpy stochastic
        operations
    :return: mappings from time series IDs to forecasts, metrics and
        model configs
    """
    relative_weights_acc, relative_weights_stb = _get_weights(weight_type, nsimulations)
    # Record forecast results.
    id_to_forecast_ensembles = {}
    # Record metrics.
    id_to_metrics = {}
    # Record model config.
    id_to_model_configs = {}
    # Seed random seeds.
    set_torch_seed(seed)
    for specific_id in selected_ids:
        filtered_data = data.filter(lambda record: record["id"] == specific_id)
        if len(filtered_data) > 0:
            record = filtered_data[0]
            targets = record["target"]
            length = len(targets)
            train_end_idx = int(length * train_ratio)
            train_ts, validate_ts = (
                targets[:train_end_idx],
                targets[train_end_idx + 1 :],
            )
            # Find model config using auto-arima.
            auto_arima_model = pmdarima.auto_arima(
                train_ts,
                start_p=0,
                max_p=3,  # AR order range
                start_q=0,
                max_q=0,  # MA order range
                d=None,  # Auto-determine differencing
                seasonal=True,  # Consider seasonal ARIMA
                m=24,  # Seasonal period
                start_P=0,
                max_P=2,  # Seasonal AR
                start_Q=0,
                max_Q=0,  # Seasonal MA
                D=None,  # Seasonal differencing
                information_criterion="aic",  # or 'bic', 'aicc'
                stepwise=True,  # Faster stepwise search
                trace=True,  # Print search progress
                error_action="ignore",
                suppress_warnings=True,
            )
            p, d, q = auto_arima_model.order
            P, D, Q, m = auto_arima_model.seasonal_order
            model_config = {
                "p": p,
                "d": d,
                "q": q,
                "P": P,
                "D": D,
                "Q": Q,
                "m": m,
            }
            # Run experiment wrapper.
            (
                train_result_default_metric,
                validate_result_default_metric,
                train_result_custom_metric,
                validate_result_custom_metric,
                forecast_ensemble_default,
                forecast_ensemble_custom,
            ) = train_ARIMA_using_two_objectives(
                model_config,
                train_ts,
                validate_ts,
                nsimulations=nsimulations,
                nrepetitions=nrepetitions,
                relative_weights_acc=relative_weights_acc,
                relative_weights_stb=relative_weights_stb,
                mult_stability=mult_stability,
                min_epochs=200,
                max_epochs=3000,
            )
            _LOG.info(f"Finished running on {specific_id}.")
            # Package results.
            id_to_forecast_ensembles[specific_id] = {
                "default": forecast_ensemble_default,
                "custom": forecast_ensemble_custom.detach().cpu(),
            }
            id_to_metrics[specific_id] = {
                "default_train": train_result_default_metric,
                "default_validate": validate_result_default_metric,
                "custom_train": train_result_custom_metric,
                "custom_validate": validate_result_custom_metric,
            }
            id_to_model_configs[specific_id] = model_config
    _LOG.info(f"Completed running on {len(selected_ids)} time series.")
    return id_to_forecast_ensembles, id_to_metrics, id_to_model_configs

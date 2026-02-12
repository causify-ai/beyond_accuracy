"""
Define the class of differentiable ARIMA models and training using gradient
descent.

Import as:

import core_pytorch.forecast_stability.differentiable_arima as cpfsdiar
"""

from typing import Optional

import torch
import torch.nn as nn


def torch_convolve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Convolve two 1D tensors (polynomial multiplication).

    Equivalent to np.convolve(a, b, mode='full').

    :param a: first tensor
    :param b: second tensor
    :return: convolution result
    """
    # Flip b for convolution.
    return torch.nn.functional.conv1d(
        a.unsqueeze(0).unsqueeze(0),
        b.flip(0).unsqueeze(0).unsqueeze(0),
        padding=len(b) - 1,
    ).squeeze()


def expand_ar_polynomial_torch(
        phi: torch.Tensor, 
        Phi: torch.Tensor, 
        s: int, 
        d: int, 
        D: int, 
        device: str,
    ) -> torch.Tensor:
    """
    Expand (1-L)^d (1-L^s)^D Phi(L) Phi_s(L^s) to get AR coefficients.

    PyTorch version that maintains gradients.

    :param phi: non-seasonal AR coefficients [phi_1, ..., phi_p]
    :param Phi: seasonal AR coefficients [Phi_1, ..., Phi_P]
    :param s: seasonal period
    :param d: non-seasonal differencing order
    :param D: seasonal differencing order
    :param device: torch device
    :return: AR coefficients tensor [a_1, a_2, ..., a_K]
    """
    # Start with polynomial = 1.
    result = torch.tensor([1.0], device=device)
    # Multiply by (1 - L)^d.
    if d > 0:
        diff_poly = torch.tensor([1.0, -1.0], device=device)
        for _ in range(d):
            result = torch_convolve(result, diff_poly)
    # Multiply by (1 - L^s)^D.
    if D > 0 and s > 0:
        seasonal_diff_poly = torch.zeros(s + 1, device=device)
        seasonal_diff_poly[0] = 1.0
        seasonal_diff_poly[s] = -1.0
        for _ in range(D):
            result = torch_convolve(result, seasonal_diff_poly)
    # Multiply by Phi(L) = 1 - phi_1*L - phi_2*L^2 - ...
    if len(phi) > 0:
        ar_poly = torch.zeros(len(phi) + 1, device=device)
        ar_poly[0] = 1.0
        ar_poly[1:] = -phi
        result = torch_convolve(result, ar_poly)
    # Multiply by Phi_s(L^s) = 1 - Phi_1*L^s - Phi_2*L^(2s) - ...
    if len(Phi) > 0 and s > 0:
        max_seasonal_lag = len(Phi) * s
        seasonal_ar_poly = torch.zeros(max_seasonal_lag + 1, device=device)
        seasonal_ar_poly[0] = 1.0
        for i, coef in enumerate(Phi):
            seasonal_ar_poly[(i + 1) * s] = -coef
        result = torch_convolve(result, seasonal_ar_poly)
    # Result is [1, -a_1, -a_2, ..., -a_K].
    # We want [a_1, a_2, ..., a_K].
    if len(result) > 1:
        return -result[1:]
    else:
        return torch.tensor([], device=device)


def expand_ma_polynomial_torch(
        theta: torch.Tensor, 
        Theta: torch.Tensor, 
        s: int, 
        device: str,
    ) -> torch.Tensor:
    """
    Expand Theta(L) Theta_s(L^s) to get MA coefficients.

    PyTorch version that maintains gradients.

    :param theta: torch tensor of non-seasonal MA coefficients [theta_1, ..., theta_q]
    :param Theta: torch tensor of seasonal MA coefficients [Theta_1, ..., Theta_Q]
    :param s: seasonal period
    :param device: torch device
    :return: MA coefficients tensor [b_1, b_2, ..., b_K]
    """
    # Start with polynomial = 1.
    result = torch.tensor([1.0], device=device)
    # Multiply by Theta(L) = 1 + theta_1*L + theta_2*L^2 + ...
    if len(theta) > 0:
        ma_poly = torch.zeros(len(theta) + 1, device=device)
        ma_poly[0] = 1.0
        ma_poly[1:] = theta
        result = torch_convolve(result, ma_poly)
    # Multiply by Theta_s(L^s) = 1 + Theta_1*L^s + Theta_2*L^(2s) + ...
    if len(Theta) > 0 and s > 0:
        max_seasonal_lag = len(Theta) * s
        seasonal_ma_poly = torch.zeros(max_seasonal_lag + 1, device=device)
        seasonal_ma_poly[0] = 1.0
        for i, coef in enumerate(Theta):
            seasonal_ma_poly[(i + 1) * s] = coef
        result = torch_convolve(result, seasonal_ma_poly)
    # Result is [1, b_1, b_2, ..., b_K].
    # We want [b_1, b_2, ..., b_K].
    if len(result) > 1:
        return result[1:]
    else:
        return torch.tensor([], device=device)


# #############################################################################
# DifferentiableARIMA
# #############################################################################


class DifferentiableARIMA(nn.Module):
    """
    ARIMA model with learnable parameters including seasonal components.

    Uses mean-centered formulation to match statsmodels:
    Phi(L) (y_t - const) = Theta(L) epsilon_t

    Where const is the unconditional mean of the (differenced) series.
    This matches statsmodels' naming convention.
    """

    def __init__(
        self,
        p: int,
        d: int,
        q: int,
        P: int = 0,
        D: int = 0,
        Q: int = 0,
        m: int = 0,
        with_intercept: bool = True,
    ):
        """
        Initialize the differentiable ARIMA model.

        :param p: non-seasonal AR order
        :param d: non-seasonal differencing order
        :param q: non-seasonal MA order
        :param P: seasonal AR order
        :param D: seasonal differencing order
        :param Q: seasonal MA order
        :param m: seasonal period (e.g., 24 for hourly data with daily
            seasonality)
        :param with_intercept: if True, const is a learnable parameter;
            if False, const is fixed at 0
        """
        super().__init__()
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.m = m
        self.with_intercept = with_intercept
        # Set non-seasonal learnable parameters.
        if p > 0:
            self.ar_params = nn.Parameter(torch.ones(p) * 0.5)
        if q > 0:
            self.ma_params = nn.Parameter(torch.randn(q) * 0.1)
        # Set seasonal learnable parameters.
        if P > 0:
            self.seasonal_ar_params = nn.Parameter(torch.ones(P) * 0.5)
        if Q > 0:
            self.seasonal_ma_params = nn.Parameter(torch.randn(Q) * 0.1)
        # Set const/intercept parameter.
        if with_intercept:
            self.const = nn.Parameter(torch.zeros(1))
        else:
            # Register as buffer so it won't be optimized).
            self.register_buffer("const", torch.zeros(1))
        # We don't learn sigma anymore.
        # TODO(Chutian): Review the simulation and remove sigma init if redundant.
        self.log_sigma = nn.Parameter(torch.zeros(1))
        # Cache expanded coefficients.
        self._update_min_history()

    @property
    def sigma(self):
        """
        Get the noise standard deviation.

        :return: sigma = exp(log_sigma)
        """
        return torch.exp(self.log_sigma)

    @property
    def intercept(self):
        """
        Compute the intercept term for the expanded form.

        For mean-centered form: (y_t - const) = phi_1*(y_{t-1} - const) + ... + epsilon_t
        Expanding: y_t = const*(1 - phi_1 - phi_2 - ... - phi_p - Phi_1 - ... - Phi_P + interactions)
                        + phi_1*y_{t-1} + ... + epsilon_t

        So: intercept = const * (1 - sum of AR coefficients).

        If with_intercept=False, const=0, so intercept=0.

        :return: intercept value
        """
        ar_coefs = self._get_expanded_ar_coefs()
        if len(ar_coefs) > 0:
            return self.const * (1.0 - torch.sum(ar_coefs))
        else:
            # Pure MA or white noise model.
            return self.const

    def compute_residuals(self, y: torch.Tensor):
        """
        Compute one-step-ahead residuals for the entire series.

        This method computes ε_t = y_t - ŷ_t where ŷ_t is the one-step-ahead
        point forecast using AR and MA components.

        :param y: observed time series (1D tensor)
        :return: residuals (1D tensor, same length as y)
        """
        device = y.device
        n = len(y)
        ar_coefs = self._get_expanded_ar_coefs()
        ma_coefs = self._get_expanded_ma_coefs()
        max_ar_lag = len(ar_coefs) if len(ar_coefs) > 0 else 0
        max_ma_lag = len(ma_coefs) if len(ma_coefs) > 0 else 0
        # Initialize residuals to zero (CSS approach).
        residuals = torch.zeros(n, device=device)
        # Start computing from where we have enough history.
        start_idx = max(max_ar_lag, max_ma_lag, 1)
        for t in range(start_idx, n):
            # Compute one-step-ahead point forecast.
            forecast = self.intercept
            # Add AR component.
            if max_ar_lag > 0:
                ar_contrib = torch.sum(ar_coefs.flip(0) * y[t - max_ar_lag : t])
                forecast = forecast + ar_contrib
            # Add MA component (using previously computed residuals).
            if max_ma_lag > 0:
                ma_contrib = torch.sum(
                    ma_coefs.flip(0) * residuals[t - max_ma_lag : t]
                )
                forecast = forecast + ma_contrib
            # Compute residual.
            residuals[t] = y[t] - forecast
        return residuals

    def simulate_forward(
        self,
        y_past: torch.Tensor,
        nsimulations: int,
        repetitions: int,
        residuals_past: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Simulate ARIMA forward using mean-centered formulation.

        :param y_past: past observations (1D tensor)
        :param nsimulations: number of steps ahead
        :param repetitions: number of sample paths (will all be
            identical for point forecasts)
        :param residuals_past: optional pre-computed residuals for MA
            initialization; if None, will compute from y_past (slower
            but accurate)
        :return: simulation of samples with shape (nsimulations,
            repetitions)
        """
        batch_size = repetitions
        device = y_past.device
        # Get expanded coefficients.
        ar_coefs = self._get_expanded_ar_coefs()
        ma_coefs = self._get_expanded_ma_coefs()
        # Determine required history length.
        max_ar_lag = len(ar_coefs) if len(ar_coefs) > 0 else 0
        max_ma_lag = len(ma_coefs) if len(ma_coefs) > 0 else 0
        # If residuals not provided and MA is needed, compute them.
        if residuals_past is None and max_ma_lag > 0:
            residuals_full = self.compute_residuals(y_past)
            residuals_past = (
                residuals_full[-max_ma_lag:]
                if len(residuals_full) >= max_ma_lag
                else residuals_full
            )
        elif residuals_past is None:
            residuals_past = torch.tensor([], device=device)
        # Initialize history for AR.
        if max_ar_lag > 0:
            n_history = min(len(y_past), max_ar_lag)
            if n_history < max_ar_lag:
                # Pad with const (not zeros!) for missing history.
                history = torch.cat(
                    [
                        self.const.expand(batch_size, max_ar_lag - n_history),
                        y_past.unsqueeze(0).repeat(batch_size, 1),
                    ],
                    dim=1,
                )
            else:
                # Use last max_ar_lag values.
                history = y_past[-max_ar_lag:].unsqueeze(0).repeat(batch_size, 1)
        else:
            history = torch.zeros(batch_size, 1, device=device)
        # Initialize error history for MA components.
        if max_ma_lag > 0:
            n_res_history = min(len(residuals_past), max_ma_lag)
            if n_res_history < max_ma_lag:
                # Pad with zeros for missing residual history.
                error_history = torch.cat(
                    [
                        torch.zeros(
                            batch_size, max_ma_lag - n_res_history, device=device
                        ),
                        residuals_past.unsqueeze(0).repeat(batch_size, 1),
                    ],
                    dim=1,
                )
            else:
                # Use last max_ma_lag residuals.
                error_history = (
                    residuals_past[-max_ma_lag:]
                    .unsqueeze(0)
                    .repeat(batch_size, 1)
                )
        else:
            error_history = torch.zeros(batch_size, 1, device=device)
        samples = []
        for t in range(nsimulations):
            # Start with the intercept.
            forecast = self.intercept.expand(batch_size)
            # Add AR component (using expanded coefficients).
            if max_ar_lag > 0:
                ar_contribution = torch.sum(
                    ar_coefs.flip(0) * history[:, -max_ar_lag:], dim=1
                )
                forecast = forecast + ar_contribution
            # Add MA component (using expanded coefficients).
            if max_ma_lag > 0:
                ma_contribution = torch.sum(
                    ma_coefs.flip(0) * error_history[:, -max_ma_lag:], dim=1
                )
                forecast = forecast + ma_contribution
            # Generate next value (point forecast, no noise).
            y_next = forecast
            # Update histories.
            if max_ar_lag > 0:
                history = torch.cat([history[:, 1:], y_next.unsqueeze(1)], dim=1)
            # For future steps: E[ε_{t+h}] = 0.
            error_history = torch.cat(
                [error_history[:, 1:], torch.zeros(batch_size, 1, device=device)],
                dim=1,
            )
            samples.append(y_next)
        # Stack into shape (nsimulations, repetitions).
        samples = torch.stack(samples, dim=0)
        return samples

    def rolling_forecast_batched(
        self,
        y: torch.Tensor,
        nsimulations: int,
        repetitions: int,
        last_forecast_origin: Optional[int] = None,
        batch_size: int = 64,
        residuals_cached: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate rolling forecasts with batching.

        :param y: training time series data
        :param nsimulations: number of steps to forecast
        :param repetitions: number of sample paths per forecast origin
        :param last_forecast_origin: index of the last forecast origin
        :param batch_size: number of forecast origins to process
            simultaneously
        :param residuals_cached: optional pre-computed residuals for
            entire series; if None, will compute on-the-fly (slower)
        :return: forecast_ensemble of shape (n_forecasts, nsimulations,
            repetitions)
        """
        device = y.device
        if not last_forecast_origin:
            # Set the last origin such that its forecast does not exceed the data length.
            last_forecast_origin = len(y) - nsimulations - 1
        # Get expanded coefficients to determine min history.
        ar_coefs = self._get_expanded_ar_coefs()
        ma_coefs = self._get_expanded_ma_coefs()
        max_ar_lag = len(ar_coefs) if len(ar_coefs) > 0 else 0
        max_ma_lag = len(ma_coefs) if len(ma_coefs) > 0 else 0
        min_history = self.min_history
        # If MA is being used and residuals not provided, compute them.
        if residuals_cached is None and max_ma_lag > 0:
            residuals_cached = self.compute_residuals(y)
        # Collect valid forecast origins.
        # Set the first origin such that its next timestamp has min_history many historical values available.
        valid_origins = [
            i for i in range(last_forecast_origin + 1) if i >= min_history - 1
        ]
        if len(valid_origins) == 0:
            return torch.zeros(0, nsimulations, repetitions, device=device)
        all_forecasts = []
        # Process in batches.
        for batch_start in range(0, len(valid_origins), batch_size):
            batch_end = min(batch_start + batch_size, len(valid_origins))
            batch_origins = valid_origins[batch_start:batch_end]
            # Stack histories for batch.
            batch_histories = []
            batch_residuals = [] if max_ma_lag > 0 else None
            for origin in batch_origins:
                y_past = y[: origin + 1]
                if len(y_past) >= max_ar_lag:
                    history = (
                        y_past[-max_ar_lag:] if max_ar_lag > 0 else y_past[-1:]
                    )
                else:
                    pad_size = max(max_ar_lag - len(y_past), 0)
                    # Pad with const, not zeros.
                    history = torch.cat([self.const.expand(pad_size), y_past])
                batch_histories.append(history)
                # Extract residuals for this origin if using MA.
                if max_ma_lag > 0:
                    if residuals_cached is not None:
                        res_past = residuals_cached[
                            max(0, origin - max_ma_lag + 1) : origin + 1
                        ]
                        if len(res_past) < max_ma_lag:
                            # Pad with zeros if not enough residual history.
                            res_past = torch.cat(
                                [
                                    torch.zeros(
                                        max_ma_lag - len(res_past), device=device
                                    ),
                                    res_past,
                                ]
                            )
                        batch_residuals.append(res_past[-max_ma_lag:])
                    else:
                        # Should not happen if residuals_cached was computed above.
                        batch_residuals.append(
                            torch.zeros(max_ma_lag, device=device)
                        )
            batch_histories = torch.stack(batch_histories)
            if batch_residuals is not None:
                batch_residuals = torch.stack(batch_residuals)
            # Simulate forward for entire batch.
            batch_samples = self.simulate_forward_batched(
                batch_histories,
                nsimulations,
                repetitions,
                batch_residuals=batch_residuals,
            )
            all_forecasts.append(batch_samples)
        # Concatenate all batches.
        forecast_ensemble = torch.cat(all_forecasts, dim=0)
        return forecast_ensemble

    def simulate_forward_batched(
        self,
        batch_histories: torch.Tensor,
        nsimulations: int,
        repetitions: int,
        batch_residuals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Simulate forward for a batch of starting histories.

        :param batch_histories: histories for each origin of shape
            (batch_size, max_lag)
        :param nsimulations: number of steps ahead
        :param repetitions: number of sample paths per origin
        :param batch_residuals: optional residual histories of shape
            (batch_size, max_ma_lag)
        :return: batched samples of forward simulation, shape
            (batch_size, nsimulations, repetitions)
        """
        batch_size = batch_histories.shape[0]
        device = batch_histories.device
        # Get expanded coefficients.
        ar_coefs = self._get_expanded_ar_coefs()
        ma_coefs = self._get_expanded_ma_coefs()
        max_ar_lag = len(ar_coefs) if len(ar_coefs) > 0 else 0
        max_ma_lag = len(ma_coefs) if len(ma_coefs) > 0 else 0
        # Expand histories to include repetitions: (batch_size * repetitions, max_lag).
        history = batch_histories.unsqueeze(1).repeat(1, repetitions, 1)
        history = history.reshape(batch_size * repetitions, -1)
        # Initialize error history.
        if max_ma_lag > 0 and batch_residuals is not None:
            error_history = batch_residuals.unsqueeze(1).repeat(1, repetitions, 1)
            error_history = error_history.reshape(batch_size * repetitions, -1)
        else:
            error_history = torch.zeros(
                batch_size * repetitions, max(max_ma_lag, 1), device=device
            )
        all_samples = []
        for t in range(nsimulations):
            # Start with the intercept (intercept = const * (1 - sum(AR coefficients))).
            forecast = self.intercept.expand(batch_size * repetitions)
            # Add AR component (using expanded coefficients).
            if max_ar_lag > 0:
                ar_contribution = torch.sum(
                    ar_coefs.flip(0) * history[:, -max_ar_lag:], dim=1
                )
                forecast = forecast + ar_contribution
            # Add MA component (using expanded coefficients).
            if max_ma_lag > 0:
                ma_contribution = torch.sum(
                    ma_coefs.flip(0) * error_history[:, -max_ma_lag:], dim=1
                )
                forecast = forecast + ma_contribution
            # Generate next values (point forecast, no noise).
            y_next = forecast
            # Update histories.
            if max_ar_lag > 0:
                history = torch.cat([history[:, 1:], y_next.unsqueeze(1)], dim=1)
            # For future steps: E[ε_{t+h}] = 0.
            error_history = torch.cat(
                [
                    error_history[:, 1:],
                    torch.zeros(batch_size * repetitions, 1, device=device),
                ],
                dim=1,
            )
            all_samples.append(y_next)
        # Stack and reshape: (batch_size, nsimulations, repetitions).
        all_samples = torch.stack(all_samples, dim=0)
        all_samples = all_samples.reshape(nsimulations, batch_size, repetitions)
        all_samples = all_samples.permute(1, 0, 2)
        return all_samples

    def _update_min_history(self) -> int:
        """
        Calculate minimum history based on expanded polynomial.
        """
        with torch.no_grad():
            ar_coefs_full = self._get_expanded_ar_coefs()
            self.min_history = len(ar_coefs_full) if len(ar_coefs_full) > 0 else 1

    def _get_expanded_ar_coefs(self, include_differencing: bool = True) -> torch.Tensor:
        """
        Get expanded AR coefficients as a torch tensor with gradients.

        :param include_differencing: if True, include (1-L)^d (1-L^s)^D
            terms
        :return: expanded AR coefficients
        """
        # Represent autoregressive polynomials as the corresponding coefficient tensors.
        phi = (
            self.ar_params
            if self.p > 0
            else torch.tensor([], device=self.const.device)
        )
        Phi = (
            self.seasonal_ar_params
            if self.P > 0
            else torch.tensor([], device=self.const.device)
        )
        # Expand polynomial.
        if include_differencing:
            ar_coefs = expand_ar_polynomial_torch(
                phi, Phi, self.m, self.d, self.D, self.const.device
            )
        else:
            ar_coefs = expand_ar_polynomial_torch(
                phi, Phi, self.m, 0, 0, self.const.device
            )
        return ar_coefs

    def _get_expanded_ma_coefs(self) -> torch.Tensor:
        """
        Get expanded MA coefficients as a torch tensor with gradients.

        :return: expanded MA coefficients
        """
        # Get parameters (keeping gradients).
        theta = (
            self.ma_params
            if self.q > 0
            else torch.tensor([], device=self.const.device)
        )
        Theta = (
            self.seasonal_ma_params
            if self.Q > 0
            else torch.tensor([], device=self.const.device)
        )
        # Expand polynomial (fully differentiable).
        ma_coefs = expand_ma_polynomial_torch(
            theta, Theta, self.m, self.const.device
        )
        return ma_coefs

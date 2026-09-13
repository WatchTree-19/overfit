"""Probabilistic and Deflated Sharpe Ratios.

PSR (Bailey & Lopez de Prado 2012): the probability that the true Sharpe
exceeds a benchmark SR*, given the estimated Sharpe, sample length and the
higher moments of returns.

DSR (Bailey & Lopez de Prado 2014): PSR evaluated against the Sharpe you
would expect the BEST of N unskilled trials to show. A backtest selected
from N trials must clear that bar before its Sharpe means anything, which
is the whole discipline the multiple-testing literature asks for and
backtesting practice ignores.

All Sharpe ratios here are per-period (not annualised); annualise inputs
consistently before calling.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329


def sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def psr(sr: float, sr_benchmark: float, n_obs: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """P(true SR > sr_benchmark) given estimated `sr` over `n_obs` periods."""
    if n_obs < 2:
        return float("nan")
    denom = np.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2)
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    z = (sr - sr_benchmark) * np.sqrt(n_obs - 1.0) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, var_sharpe: float) -> float:
    """E[max SR] across n unskilled trials whose SR estimates have variance
    var_sharpe (Bailey & Lopez de Prado 2014, extreme-value approximation)."""
    if n_trials <= 1:
        return 0.0
    e = (1.0 - EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n_trials) \
        + EULER_GAMMA * norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_sharpe) * e)


def deflated_sharpe(
    returns: np.ndarray,
    n_trials: int,
    var_sharpe_across_trials: float,
) -> dict:
    """DSR of the SELECTED strategy given how many configurations were tried.

    var_sharpe_across_trials is the variance of the SR estimates across the
    trials (compute it from the trial matrix; it is NOT the variance of
    returns). Returns the estimated SR, the deflation benchmark SR*, and DSR.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    sr = sharpe(r)
    sk = 0.0 if r.std(ddof=1) == 0 else float(
        ((r - r.mean()) ** 3).mean() / r.std(ddof=0) ** 3)
    ku = 3.0 if r.std(ddof=1) == 0 else float(
        ((r - r.mean()) ** 4).mean() / r.std(ddof=0) ** 4)
    sr_star = expected_max_sharpe(n_trials, var_sharpe_across_trials)
    return {
        "sharpe": sr,
        "sr_star": sr_star,
        "dsr": psr(sr, sr_star, len(r), sk, ku),
        "n_obs": len(r),
        "skew": sk,
        "kurtosis": ku,
    }

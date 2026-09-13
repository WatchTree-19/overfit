import numpy as np
import pytest
from scipy.stats import norm

from overfit import cscv_pbo, deflated_sharpe, expected_max_sharpe, psr, sharpe


def test_psr_gaussian_closed_form():
    # With zero skew and kurtosis 3, the estimator-variance denominator is
    # sqrt(1 + sr^2/2), so PSR(0, sr) = Phi(sr sqrt(T-1) / sqrt(1 + sr^2/2)).
    sr, T = 0.1, 101
    want = float(norm.cdf(sr * np.sqrt(T - 1) / np.sqrt(1 + sr**2 / 2)))
    assert psr(sr, 0.0, T) == pytest.approx(want, abs=1e-12)


def test_expected_max_sharpe_monotone_in_trials():
    v = 0.02
    vals = [expected_max_sharpe(n, v) for n in (2, 10, 100, 1000)]
    assert all(b > a for a, b in zip(vals, vals[1:]))
    assert expected_max_sharpe(1, v) == 0.0


def test_dsr_punishes_selection():
    rng = np.random.default_rng(0)
    T, N = 1000, 200
    M = rng.normal(0, 0.01, size=(T, N))          # pure noise trials
    srs = np.array([sharpe(M[:, j]) for j in range(N)])
    best = int(np.argmax(srs))
    out = deflated_sharpe(M[:, best], n_trials=N, var_sharpe_across_trials=float(srs.var(ddof=1)))
    # The best of 200 noise strategies has a healthy raw Sharpe...
    assert out["sharpe"] > 0.05
    # ...its naive PSR against zero looks convincing...
    assert psr(out["sharpe"], 0.0, T) > 0.95
    # ...and the DSR correctly refuses to be impressed: the max of N noise
    # trials sits AT the deflation benchmark on average, so its DSR hovers
    # near one half instead of the naive 0.95+.
    assert out["dsr"] < psr(out["sharpe"], 0.0, T) - 0.2


def test_dsr_passes_genuine_skill():
    rng = np.random.default_rng(1)
    r = rng.normal(0.002, 0.01, size=1000)        # true SR 0.2 per period
    out = deflated_sharpe(r, n_trials=200, var_sharpe_across_trials=0.001)
    assert out["dsr"] > 0.95


def test_pbo_near_half_on_noise():
    rng = np.random.default_rng(2)
    M = rng.normal(0, 0.01, size=(640, 50))
    res = cscv_pbo(M, n_blocks=8)
    assert 0.3 < res["pbo"] < 0.7
    assert res["n_combinations"] == 70


def test_pbo_low_with_true_skill():
    rng = np.random.default_rng(3)
    M = rng.normal(0, 0.01, size=(640, 50))
    M[:, 7] += 0.004                               # one genuinely skilled config
    res = cscv_pbo(M, n_blocks=8)
    assert res["pbo"] < 0.1


def test_pbo_input_validation():
    with pytest.raises(ValueError):
        cscv_pbo(np.zeros((100, 3)), n_blocks=7)
    with pytest.raises(ValueError):
        cscv_pbo(np.zeros(100))

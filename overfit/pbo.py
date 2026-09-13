"""Probability of Backtest Overfitting via CSCV.

Bailey, Borwein, Lopez de Prado and Zhu (2015): split the sample into S
blocks; for every balanced combination of S/2 blocks as in-sample, pick the
configuration with the best in-sample Sharpe and observe its RELATIVE RANK
out of sample. If selection carried no information, that rank is uniform;
PBO is the fraction of combinations in which the in-sample winner performs
below the out-of-sample median. Pure noise gives PBO near one half; a truly
skilled configuration drives it towards zero; heavy selection over noise
drives OOS logits negative.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from overfit.dsr import sharpe


def cscv_pbo(trial_returns: np.ndarray, n_blocks: int = 16, metric=sharpe) -> dict:
    """trial_returns: (T, N) matrix, one column per strategy configuration.

    Returns PBO, the per-combination logits, and the mean OOS rank of the
    in-sample winner. n_blocks must be even; T is truncated to a multiple of
    n_blocks.
    """
    M = np.asarray(trial_returns, dtype=float)
    if M.ndim != 2:
        raise ValueError("trial_returns must be (T, N)")
    T, N = M.shape
    if n_blocks % 2:
        raise ValueError("n_blocks must be even")
    rows = (T // n_blocks) * n_blocks
    if rows < n_blocks:
        raise ValueError("not enough observations for the requested blocks")
    blocks = np.array_split(M[:rows], n_blocks)

    logits = []
    ranks = []
    idx = range(n_blocks)
    for insample in combinations(idx, n_blocks // 2):
        ins = np.vstack([blocks[i] for i in insample])
        oos = np.vstack([blocks[i] for i in idx if i not in insample])
        is_perf = np.array([metric(ins[:, j]) for j in range(N)])
        oos_perf = np.array([metric(oos[:, j]) for j in range(N)])
        star = int(np.argmax(is_perf))
        # relative rank of the winner's OOS performance in (0, 1)
        omega = (np.sum(oos_perf <= oos_perf[star])) / (N + 1.0)
        omega = min(max(omega, 1e-9), 1 - 1e-9)
        logits.append(np.log(omega / (1.0 - omega)))
        ranks.append(omega)

    logits = np.array(logits)
    return {
        "pbo": float(np.mean(logits <= 0.0)),
        "mean_oos_rank": float(np.mean(ranks)),
        "n_combinations": len(logits),
        "logits": logits,
    }

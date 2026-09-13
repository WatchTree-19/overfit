"""The two instruments on a synthetic parameter sweep. No private data.

    pip install -e .
    python example.py

The point of the example is the DISAGREEMENT between them. A family of
configurations can carry genuine signal, so the selected one clears the
deflation bar, whilst the CHOICE of configuration within that family carries
none, so PBO sits near a half. One number cannot say both things.
"""

import numpy as np

from overfit import cscv_pbo, deflated_sharpe, expected_max_sharpe, sharpe


def noise_sweep(n_obs=2000, n_trials=60, seed=0):
    """Sixty configurations, none of which has any edge."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.01, (n_obs, n_trials))


def signal_plus_noise_sweep(n_obs=2000, n_trials=60, seed=1, edge=0.0006):
    """A shared drift every configuration inherits, plus idiosyncratic noise.

    Every column has the same true edge, so choosing between columns is
    choosing between identical strategies and can carry no information.
    """
    rng = np.random.default_rng(seed)
    common = rng.normal(edge, 0.008, (n_obs, 1))
    return common + rng.normal(0.0, 0.004, (n_obs, n_trials))


def report(name, trials):
    srs = np.array([sharpe(trials[:, j]) for j in range(trials.shape[1])])
    winner = trials[:, int(np.argmax(srs))]
    d = deflated_sharpe(winner, n_trials=trials.shape[1],
                        var_sharpe_across_trials=float(srs.var(ddof=1)))
    p = cscv_pbo(trials, n_blocks=10)
    print("=== %s" % name)
    print("  configurations tried        %d" % trials.shape[1])
    print("  best in-sample Sharpe       %+.4f per period" % d["sharpe"])
    print("  deflation bar SR*           %+.4f" % d["sr_star"])
    print("  deflated Sharpe (DSR)       %.4f" % d["dsr"])
    print("  PBO                         %.3f" % p["pbo"])
    print("  mean OOS rank of winner     %.3f" % p["mean_oos_rank"])
    print()


print("E[max Sharpe] across N unskilled trials, per-period SR variance 0.0009:")
for n in (1, 5, 20, 100, 1000):
    print("  N = %5d  ->  SR* = %+.4f" % (n, expected_max_sharpe(n, 0.0009)))
print()

report("pure noise, sixty configurations", noise_sweep())
report("shared edge, sixty configurations", signal_plus_noise_sweep())

print("Read it this way. On pure noise the winner's Sharpe does not clear the")
print("bar the best of sixty coin flips would show, so the DSR is around a")
print("half, and PBO comes out ABOVE a half: the in-sample winner lands below")
print("the out-of-sample median more often than not, which is selection")
print("actively misleading rather than merely failing. On the second sweep")
print("the family carries a real edge, the DSR says so, and PBO falls well")
print("below a half because any configuration inherits that edge.")
print()
print("The pair is the point. DSR asks whether the winning number survives")
print("the search that produced it. PBO asks whether the search itself picked")
print("anything. A single headline Sharpe answers neither.")

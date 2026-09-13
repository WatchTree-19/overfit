# overfit

Two instruments for the question every backtest dodges: **is this Sharpe
ratio the reward for skill, or the reward for searching?**

A strategy that was chosen from sixty configurations is not the same object
as a strategy that was specified in advance, and its Sharpe ratio does not
mean the same thing. This library implements the two published tests that
say so, and it implements them together, because they answer different
questions and routinely disagree.

## The two questions

**Did the winning number survive the search that produced it?** The Deflated
Sharpe Ratio (Bailey and Lopez de Prado) computes the Sharpe the *best of N
unskilled trials* would be expected to show, then asks whether the selected
strategy clears that bar. `expected_max_sharpe(n_trials, var_sharpe)` is that
bar, and the bar rises with the number of trials whether or not you report
how many you ran.

**Did the search pick anything at all?** The Probability of Backtest
Overfitting via CSCV (Bailey, Borwein, Lopez de Prado and Zhu) splits the
sample into blocks, repeatedly takes half as in-sample, picks the in-sample
winner, and records where it lands out of sample. If selection carried no
information, that rank is uniform and PBO is near a half. Above a half means
the in-sample winner tends to land *below* the out-of-sample median, which is
selection actively misleading rather than merely failing.

## Install

    pip install -e .

Python 3.10 or newer, numpy and scipy. Nothing else.

## Use

```python
import numpy as np
from overfit import sharpe, deflated_sharpe, cscv_pbo

trials = np.loadtxt("trial_returns.csv", delimiter=",")   # (T, N), one column per config
srs = np.array([sharpe(trials[:, j]) for j in range(trials.shape[1])])
winner = trials[:, int(np.argmax(srs))]

d = deflated_sharpe(winner,
                    n_trials=trials.shape[1],
                    var_sharpe_across_trials=float(srs.var(ddof=1)))
print(d["sharpe"], d["sr_star"], d["dsr"])

print(cscv_pbo(trials, n_blocks=10)["pbo"])
```

`var_sharpe_across_trials` is the variance of the Sharpe *estimates across
your trials*, not the variance of returns. Getting that wrong is the usual
way to compute a deflation bar that is far too low.

All Sharpe ratios are per-period. Annualise consistently before calling, or
not at all.

## What the example shows

`python example.py` runs both instruments over two synthetic sweeps of sixty
configurations each:

    === pure noise, sixty configurations
      best in-sample Sharpe       +0.0418 per period
      deflation bar SR*           +0.0462
      deflated Sharpe (DSR)       0.4213
      PBO                         0.675

    === shared edge, sixty configurations
      best in-sample Sharpe       +0.0796 per period
      deflation bar SR*           +0.0268
      deflated Sharpe (DSR)       0.9909
      PBO                         0.337

On pure noise the winner's Sharpe does not clear the bar the best of sixty
coin flips would show, and PBO comes out above a half. On the second sweep
every configuration inherits one genuine edge: the DSR recognises it, and PBO
falls because any choice inherits the same edge.

The example also prints how the bar moves with the number of trials, which is
the number most backtest reports omit:

    N =     1  ->  SR* = +0.0000
    N =    20  ->  SR* = +0.0570
    N =  1000  ->  SR* = +0.0977

## Tests

    python -m pytest tests -q

They pin the Gaussian PSR closed form, monotonicity of the deflation bar in
the number of trials, the DSR discounting a best-of-noise selection, and PBO
near one half on noise against near zero with genuine skill.

## References

Bailey, D. and Lopez de Prado, M. (2012). The Sharpe Ratio Efficient
Frontier. *Journal of Risk*.

Bailey, D. and Lopez de Prado, M. (2014). The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality.
*Journal of Portfolio Management*.

Bailey, D., Borwein, J., Lopez de Prado, M. and Zhu, Q. (2015). The
Probability of Backtest Overfitting. *Journal of Computational Finance*.

## Licence

MIT. Sandeep Singh Rai, ORCID 0009-0001-3360-9205.

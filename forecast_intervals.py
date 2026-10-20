"""Posterior predictive intervals for next-season 3PM, and the anomaly flags built on them.

After season t a player's posterior is Beta(alpha + x, beta + n - x). Next season's
makes on n_next attempts then follow a Beta-Binomial with those parameters. A season
whose makes land outside the central interval of that distribution is flagged.

Imported by 07_export_powerbi.py, 08_build_excel.py, and the tests (snake_case name
because Python cannot import a file that starts with a digit).
"""

import numpy as np
from scipy.stats import betabinom

DEFAULT_LEVEL = 0.95


def posterior_params(x, n, alpha, beta):
    """Season-t posterior Beta parameters: (alpha + x, beta + n - x)."""
    x, n = np.asarray(x, dtype=float), np.asarray(n, dtype=float)
    return alpha + x, beta + n - x


def predictive_interval(x, n, alpha, beta, n_next, level=DEFAULT_LEVEL):
    """Central interval of next-season makes. Returns (lo, hi) arrays of makes.

    Makes are discrete, so the interval covers at least `level` of the probability,
    usually a bit more.
    """
    a, b = posterior_params(x, n, alpha, beta)
    dist = betabinom(np.asarray(n_next, dtype=int), a, b)
    tail = (1 - level) / 2
    return dist.ppf(tail), dist.ppf(1 - tail)


def predictive_pct(x, n, alpha, beta, n_next, x_next):
    """Where the actual makes fall in their predictive distribution, in [0, 1].

    Uses the mid-point PIT, P(X < x_next) + P(X = x_next) / 2, so a discrete outcome
    sitting exactly at the median scores 0.5. If the model is right these values are
    roughly uniform; a U-shaped histogram means the intervals are too narrow.
    """
    a, b = posterior_params(x, n, alpha, beta)
    dist = betabinom(np.asarray(n_next, dtype=int), a, b)
    x_next = np.asarray(x_next, dtype=float)
    return dist.cdf(x_next - 1) + 0.5 * dist.pmf(x_next)


def flag_anomalies(df, level=DEFAULT_LEVEL):
    """Add PI_LOW, PI_HIGH, PIT, IS_ANOMALY, DIRECTION to a player-pair frame.

    Needs columns FG3M, FG3A (season t), ALPHA, BETA (season t prior), and
    FG3M_NEXT, FG3A_NEXT (season t+1). PI_LOW / PI_HIGH are in makes.
    DIRECTION is "hot" above the interval, "cold" below it, else "expected".
    """
    out = df.copy()
    args = (out["FG3M"], out["FG3A"], out["ALPHA"], out["BETA"], out["FG3A_NEXT"])

    lo, hi = predictive_interval(*args, level=level)
    out["PI_LOW"] = lo
    out["PI_HIGH"] = hi
    out["PIT"] = predictive_pct(*args, out["FG3M_NEXT"])

    above = out["FG3M_NEXT"] > out["PI_HIGH"]
    below = out["FG3M_NEXT"] < out["PI_LOW"]
    out["IS_ANOMALY"] = (above | below).astype(int)
    out["DIRECTION"] = np.select([above, below], ["hot", "cold"], default="expected")
    return out

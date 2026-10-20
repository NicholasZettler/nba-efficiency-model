"""Checks on the predictive intervals: correct by construction before we use them on real data."""

import numpy as np
import pandas as pd
from scipy.stats import betabinom

from forecast_intervals import flag_anomalies, predictive_interval, predictive_pct

ALPHA, BETA = 130.0, 235.0   # close to a typical fitted season (prior mean ~.356)


def simulate(n_players=10_000, seed=0):
    """Players drawn from the model itself: fixed talent p ~ Beta, makes ~ Binomial."""
    rng = np.random.default_rng(seed)
    p = rng.beta(ALPHA, BETA, n_players)
    n = rng.integers(30, 700, n_players)
    n_next = rng.integers(30, 700, n_players)
    return pd.DataFrame({
        "FG3M": rng.binomial(n, p), "FG3A": n,
        "FG3M_NEXT": rng.binomial(n_next, p), "FG3A_NEXT": n_next,
        "ALPHA": ALPHA, "BETA": BETA,
    })


def test_interval_contains_mean():
    x, n, n_next = np.array([10, 40, 150]), np.array([40, 100, 400]), np.array([50, 200, 500])
    lo, hi = predictive_interval(x, n, ALPHA, BETA, n_next)
    mean = betabinom(n_next, ALPHA + x, BETA + n - x).mean()
    assert np.all((lo <= mean) & (mean <= hi))


def test_simulated_coverage():
    """If the data really come from the model, about 95% land inside (a bit more: discreteness)."""
    flagged = flag_anomalies(simulate())
    coverage = 1 - flagged["IS_ANOMALY"].mean()
    assert 0.945 <= coverage <= 0.975


def test_simulated_pit_is_uniform():
    """Under the model the PIT is ~Uniform(0, 1): mean 1/2, variance 1/12."""
    sim = simulate()
    pit = predictive_pct(sim["FG3M"], sim["FG3A"], ALPHA, BETA, sim["FG3A_NEXT"], sim["FG3M_NEXT"])
    assert abs(pit.mean() - 0.5) < 0.01
    assert abs(pit.var() - 1 / 12) < 0.005


def test_more_attempts_narrows_pct_interval():
    """In 3P% terms the interval shrinks as next season's attempts grow."""
    n_next = np.array([50, 200, 800])
    lo, hi = predictive_interval(np.full(3, 70), np.full(3, 200), ALPHA, BETA, n_next)
    width_pct = (hi - lo) / n_next
    assert np.all(np.diff(width_pct) < 0)


def test_direction_labels():
    df = pd.DataFrame({
        "FG3M": [70, 70, 70], "FG3A": [200, 200, 200],
        "FG3M_NEXT": [5, 70, 190], "FG3A_NEXT": [200, 200, 200],
        "ALPHA": ALPHA, "BETA": BETA,
    })
    out = flag_anomalies(df)
    assert out["DIRECTION"].tolist() == ["cold", "expected", "hot"]
    assert out["IS_ANOMALY"].tolist() == [1, 0, 1]

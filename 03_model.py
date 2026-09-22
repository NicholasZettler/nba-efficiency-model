"""Fit a Beta prior per season (empirical Bayes) and shrink each player's 3P%."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln

PROC_DIR = Path("data/processed")
IN_PATH = PROC_DIR / "player_season_3pt.csv"
OUT_PATH = PROC_DIR / "player_season_shrunk.csv"


def neg_loglik(params, x, n):
    """Negative Beta-Binomial log-likelihood. params are log(alpha), log(beta)."""
    alpha, beta = np.exp(params)
    ll = betaln(x + alpha, n - x + beta) - betaln(alpha, beta)
    return -ll.sum()


def fit_mom(x, n):
    """Method-of-moments starting values from observed rates."""
    p = x / n
    m, v = p.mean(), p.var(ddof=1)
    total = max(m * (1 - m) / v - 1, 1.0)
    return m * total, (1 - m) * total


def fit_season(x, n):
    """MLE for alpha, beta. Returns (alpha, beta)."""
    a0, b0 = fit_mom(x, n)
    res = minimize(
        neg_loglik,
        x0=np.log([a0, b0]),
        args=(x, n),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000},
    )
    if not res.success:
        print(f"  warning: optimizer did not converge ({res.message})")
    return tuple(np.exp(res.x))


def main():
    df = pd.read_csv(IN_PATH)
    print(f"loaded {len(df):,} player-seasons\n")

    print(f"{'season':>9} {'players':>8} {'alpha':>8} {'beta':>8} "
          f"{'prior mean':>11} {'a+b':>7}")
    print("-" * 56)

    pieces = []
    for season, grp in df.groupby("SEASON_YEAR"):
        x = grp["FG3M"].to_numpy(dtype=float)
        n = grp["FG3A"].to_numpy(dtype=float)

        alpha, beta = fit_season(x, n)
        prior_mean = alpha / (alpha + beta)
        strength = alpha + beta

        out = grp.copy()
        out["ALPHA"] = alpha
        out["BETA"] = beta
        out["PRIOR_MEAN"] = prior_mean
        out["FG3_PCT_SHRUNK"] = (out["FG3M"] + alpha) / (out["FG3A"] + strength)
        out["WEIGHT"] = out["FG3A"] / (out["FG3A"] + strength)
        out["SHIFT"] = out["FG3_PCT_SHRUNK"] - out["FG3_PCT_RAW"]
        pieces.append(out)

        print(f"{season:>9} {len(grp):>8,} {alpha:>8.2f} {beta:>8.2f} "
              f"{prior_mean:>11.4f} {strength:>7.1f}")

    result = pd.concat(pieces, ignore_index=True)
    result.to_csv(OUT_PATH, index=False)
    print(f"\nsaved -> {OUT_PATH}")

    latest = result[result["SEASON_YEAR"] == result["SEASON_YEAR"].max()]
    cols = ["PLAYER_NAME", "FG3M", "FG3A", "FG3_PCT_RAW", "FG3_PCT_SHRUNK", "SHIFT"]

    print("\nbiggest downward corrections (small-sample hot shooters):")
    print(latest.nsmallest(5, "SHIFT")[cols].to_string(index=False))

    print("\nbiggest upward corrections:")
    print(latest.nlargest(5, "SHIFT")[cols].to_string(index=False))

    print("\nleast moved (high volume, trusted):")
    print(latest.nlargest(5, "WEIGHT")[cols].to_string(index=False))


if __name__ == "__main__":
    main()

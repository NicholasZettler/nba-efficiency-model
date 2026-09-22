"""Do shrunk estimates predict next season better than raw percentages?"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

PROC_DIR = Path("data/processed")
IN_PATH = PROC_DIR / "player_season_shrunk.csv"
OUT_PATH = PROC_DIR / "validation_pairs.csv"


def season_start(s):
    return int(s.split("-")[0])


def build_pairs(df):
    """Join each player-season to the same player's NEXT season."""
    seasons = sorted(df["SEASON_YEAR"].unique(), key=season_start)
    pairs = []

    for a, b in zip(seasons, seasons[1:]):
        if season_start(b) != season_start(a) + 1:
            print(f"skipping {a} -> {b}: not consecutive")
            continue

        left = df[df["SEASON_YEAR"] == a][
            ["PLAYER_ID", "PLAYER_NAME", "FG3A", "FG3_PCT_RAW", "FG3_PCT_SHRUNK"]
        ]
        right = df[df["SEASON_YEAR"] == b][["PLAYER_ID", "FG3A", "FG3_PCT_RAW"]]

        merged = left.merge(right, on="PLAYER_ID", suffixes=("_T", "_NEXT"))
        merged["PAIR"] = f"{a} -> {b}"
        pairs.append(merged)

    return pd.concat(pairs, ignore_index=True)


def score(pred, actual, weights):
    """RMSE, weighted RMSE, MAE, and calibration slope/intercept."""
    err = pred - actual
    out = {
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "wRMSE": np.sqrt(np.sum(weights * err ** 2) / np.sum(weights)),
        "MAE": np.mean(np.abs(err)),
        "slope": np.nan,
        "intercept": np.nan,
        "r": np.nan,
    }
    if pred.max() > pred.min():  # a constant prediction has no slope to fit
        fit = linregress(pred, actual)
        out.update(slope=fit.slope, intercept=fit.intercept, r=fit.rvalue)
    return out


def report(df, label):
    actual = df["FG3_PCT_RAW_NEXT"].to_numpy()
    w = df["FG3A_NEXT"].to_numpy(dtype=float)

    raw = score(df["FG3_PCT_RAW_T"].to_numpy(), actual, w)
    shr = score(df["FG3_PCT_SHRUNK"].to_numpy(), actual, w)
    base = score(np.full(len(df), actual.mean()), actual, w)

    print(f"\n{label}   (n = {len(df):,})")
    print(f"{'':>10} {'RMSE':>9} {'wRMSE':>9} {'MAE':>9} {'slope':>9} {'r':>8}")
    print("-" * 58)
    for name, m in [("raw", raw), ("shrunk", shr)]:
        print(f"{name:>10} {m['RMSE']:>9.4f} {m['wRMSE']:>9.4f} "
              f"{m['MAE']:>9.4f} {m['slope']:>9.3f} {m['r']:>8.3f}")
    print(f"{'baseline':>10} {base['RMSE']:>9.4f} {base['wRMSE']:>9.4f} "
          f"{base['MAE']:>9.4f} {'--':>9} {'--':>8}")

    gain = 100 * (raw["RMSE"] - shr["RMSE"]) / raw["RMSE"]
    print(f"  RMSE improvement from shrinkage: {gain:+.2f}%")


def main():
    df = pd.read_csv(IN_PATH)
    pairs = build_pairs(df)
    pairs.to_csv(OUT_PATH, index=False)
    print(f"\nbuilt {len(pairs):,} player-season pairs -> {OUT_PATH}")

    report(pairs, "ALL PAIRS POOLED")

    print("\n" + "=" * 58)
    print("BY SEASON PAIR")
    for name, grp in pairs.groupby("PAIR"):
        report(grp, name)

    print("\n" + "=" * 58)
    print("BY SEASON-T VOLUME  (shrinkage should help most at low n)")
    bins = [0, 100, 200, 400, np.inf]
    labels = ["<100 3PA", "100-199", "200-399", "400+"]
    pairs["BUCKET"] = pd.cut(pairs["FG3A_T"], bins=bins, labels=labels)
    for name, grp in pairs.groupby("BUCKET", observed=True):
        if len(grp) >= 20:
            report(grp, str(name))


if __name__ == "__main__":
    main()

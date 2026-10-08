"""Build the ML table: season-t features -> season t+1 3P% (one row per player pair)."""

from pathlib import Path

import numpy as np
import pandas as pd

PROC_DIR = Path("data/processed")
IN_PATH = PROC_DIR / "player_season_shrunk.csv"
OUT_PATH = PROC_DIR / "ml_pairs.csv"

# Season-t columns used as model inputs.
FEATURES_T = [
    "FG3_PCT_RAW", "FG3_PCT_SHRUNK", "PRIOR_MEAN",
    "FG3A", "FG3A_PG", "FG3A_RATE",
    "FT_PCT", "FTA", "MIN_PG", "G", "AGE",
]


def season_start(s):
    return int(s.split("-")[0])


def add_previous_season(df):
    """Attach the player's season t-1 shooting, if he qualified that season."""
    prev = df[["PLAYER_ID", "START", "FG3M", "FG3A", "FG3_PCT_RAW"]].copy()
    prev["START"] += 1  # season t-1's row lines up with season t
    prev = prev.rename(columns={
        "FG3M": "FG3M_PREV", "FG3A": "FG3A_PREV", "FG3_PCT_RAW": "FG3_PCT_PREV",
    })
    df = df.merge(prev, on=["PLAYER_ID", "START"], how="left")

    df["HAS_PREV"] = df["FG3A_PREV"].notna().astype(int)
    # Two-season pooled 3P%: makes / attempts across t-1 and t (just t if no t-1).
    df["FG3_PCT_2YR"] = (
        (df["FG3M"] + df["FG3M_PREV"].fillna(0))
        / (df["FG3A"] + df["FG3A_PREV"].fillna(0))
    )
    return df


def build_pairs(df):
    """Join each player-season (t) to the same player's next season (t+1)."""
    feats = df[["PLAYER_ID", "PLAYER_NAME", "SEASON_YEAR", "START"] + FEATURES_T
               + ["FG3_PCT_PREV", "FG3A_PREV", "HAS_PREV", "FG3_PCT_2YR"]]

    nxt = df[["PLAYER_ID", "START", "SEASON_YEAR", "FG3M", "FG3A", "FG3_PCT_RAW"]].copy()
    nxt["START"] -= 1  # season t+1's row lines up with season t
    nxt = nxt.rename(columns={
        "SEASON_YEAR": "SEASON_NEXT", "FG3M": "FG3M_NEXT",
        "FG3A": "FG3A_NEXT", "FG3_PCT_RAW": "TARGET",
    })

    pairs = feats.merge(nxt, on=["PLAYER_ID", "START"], how="inner")
    pairs["PAIR"] = pairs["SEASON_YEAR"] + " -> " + pairs["SEASON_NEXT"]
    return pairs.sort_values(["START", "PLAYER_ID"]).reset_index(drop=True)


def main():
    df = pd.read_csv(IN_PATH)
    df["START"] = df["SEASON_YEAR"].map(season_start)
    print(f"loaded {len(df):,} player-seasons")

    df = add_previous_season(df)
    pairs = build_pairs(df)
    pairs.to_csv(OUT_PATH, index=False)
    print(f"built {len(pairs):,} pairs -> {OUT_PATH}\n")

    print("pairs per season:")
    print(pairs.groupby("PAIR").size().to_string())

    print("\nmissing values per feature:")
    na = pairs.isna().sum()
    print(na[na > 0].to_string() if na.any() else "  none")

    feature_cols = FEATURES_T + ["FG3_PCT_PREV", "FG3A_PREV", "HAS_PREV", "FG3_PCT_2YR"]
    corr = pairs[feature_cols + ["TARGET"]].corr()["TARGET"].drop("TARGET")
    print("\ncorrelation with next-season 3P% (strongest first):")
    print(corr.reindex(corr.abs().sort_values(ascending=False).index)
              .round(3).to_string())


if __name__ == "__main__":
    main()

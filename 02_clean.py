"""Collapse game logs to one row per player-season with shooting totals and features."""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw_v2")
AGE_DIR = RAW_DIR / "ages"
PROC_DIR = Path("data/processed")
OUT_PATH = PROC_DIR / "player_season_3pt.csv"

MIN_3PA_PER_GAME = 1.5
MIN_GAMES = 20


def load_all_seasons():
    """Read every game-log CSV and stack them into one frame."""
    files = sorted(RAW_DIR.glob("gamelogs_*.csv"))
    if not files:
        raise RuntimeError(f"No game-log CSVs in {RAW_DIR.resolve()}")

    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    print(f"loaded {len(files)} files -> {len(df):,} game rows")
    return df


def load_ages():
    """Stack the per-season age files: one row per (season, player)."""
    files = sorted(AGE_DIR.glob("ages_*.csv"))
    if not files:
        raise RuntimeError(f"No age CSVs in {AGE_DIR.resolve()}")
    ages = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    return ages.drop_duplicates(subset=["SEASON_YEAR", "PLAYER_ID"])


def safe_div(num, den):
    """num / den, with NaN where den is 0."""
    return num / den.replace(0, np.nan)


def collapse(df):
    """Game grain -> player-season grain."""
    df = df.drop_duplicates(subset=["PLAYER_ID", "GAME_ID"])

    out = (
        df.groupby(["SEASON_YEAR", "PLAYER_ID"], as_index=False)
          .agg(
              PLAYER_NAME=("PLAYER_NAME", "first"),
              G=("GAME_ID", "nunique"),
              MIN=("MIN", "sum"),
              FGM=("FGM", "sum"),
              FGA=("FGA", "sum"),
              FG3M=("FG3M", "sum"),
              FG3A=("FG3A", "sum"),
              FTM=("FTM", "sum"),
              FTA=("FTA", "sum"),
          )
    )

    out["FG3A_PG"] = out["FG3A"] / out["G"]
    out["MIN_PG"] = out["MIN"] / out["G"]
    out["FG3_PCT_RAW"] = safe_div(out["FG3M"], out["FG3A"])
    out["FT_PCT"] = safe_div(out["FTM"], out["FTA"])
    out["FG3A_RATE"] = safe_div(out["FG3A"], out["FGA"])  # share of shots that are threes
    return out


def sweep_thresholds(out):
    """Show what each 3PA/G cutoff costs you, before committing to one."""
    total_attempts = out["FG3A"].sum()
    print(f"\n{'3PA/G':>6} {'players':>9} {'med 3PA':>9} {'min 3PA':>9} {'% attempts kept':>16}")
    print("-" * 54)

    for cut in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        sub = out[(out["FG3A_PG"] >= cut) & (out["G"] >= MIN_GAMES)]
        pct = 100 * sub["FG3A"].sum() / total_attempts
        print(f"{cut:>6.1f} {len(sub):>9,} {sub['FG3A'].median():>9.0f} "
              f"{sub['FG3A'].min():>9.0f} {pct:>15.1f}%")


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_all_seasons()
    out = collapse(raw)

    ages = load_ages()
    out = out.merge(ages, on=["SEASON_YEAR", "PLAYER_ID"], how="left")
    missing_age = out["AGE"].isna().sum()
    print(f"\ncollapsed to {len(out):,} player-seasons ({missing_age} missing AGE)")
    print(out.groupby("SEASON_YEAR").size().to_string())

    sweep_thresholds(out)

    qualified = out[
        (out["FG3A_PG"] >= MIN_3PA_PER_GAME) & (out["G"] >= MIN_GAMES)
    ].copy()

    qualified.to_csv(OUT_PATH, index=False)
    print(f"\nsaved {len(qualified):,} qualified player-seasons -> {OUT_PATH}")

    latest = qualified["SEASON_YEAR"].max()
    print(f"\ntop 5 by attempts, {latest}:")
    print(
        qualified[qualified["SEASON_YEAR"] == latest]
        .nlargest(5, "FG3A")[["PLAYER_NAME", "AGE", "FG3M", "FG3A",
                              "FG3_PCT_RAW", "FT_PCT", "FG3A_RATE"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

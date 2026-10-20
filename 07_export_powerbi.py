"""Reshape the pipeline outputs into a star schema for the Power BI forecast monitor.

Two fact tables share four dimensions:
  fact_pair      one row per player pair (season t -> t+1), with anomaly flags
  fact_forecast  one row per player pair x forecast method, with errors
Both carry the dimension keys, so a slicer on season, player, or volume filters both.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from forecast_intervals import flag_anomalies

PROC_DIR = Path("data/processed")
OUT_DIR = Path("data/powerbi")

# Volume buckets match 04_validate.py and 06_ml.py.
BUCKET_EDGES = [0, 100, 200, 400, np.inf]
BUCKET_LABELS = ["<100 3PA", "100-199", "200-399", "400+"]

# key: (column holding the prediction, label, family, sort order)
METHODS = {
    "raw":    ("FG3_PCT_RAW",    "Raw 3P%",           "baseline", 1),
    "league": ("PRIOR_MEAN",     "League average",    "baseline", 2),
    "eb":     ("FG3_PCT_SHRUNK", "EB shrunk",         "bayes",    3),
    "ridge":  ("PRED_RIDGE",     "Ridge",             "ml",       4),
    "gbm":    ("PRED_GBM",       "Gradient boosting", "ml",       5),
}


def load_inputs():
    """Read the pair table, the walk-forward ML predictions, and the shrunk player-seasons."""
    pairs = pd.read_csv(PROC_DIR / "ml_pairs.csv")
    preds = pd.read_csv(PROC_DIR / "ml_predictions.csv")
    shrunk = pd.read_csv(PROC_DIR / "player_season_shrunk.csv")
    print(f"loaded {len(pairs):,} pairs, {len(preds):,} ML predictions, "
          f"{len(shrunk):,} player-seasons")
    return pairs, preds, shrunk


def build_fact_pair(pairs, preds, shrunk):
    """One row per player pair: season t and t+1 shooting, ML predictions, anomaly flags."""
    season_t = shrunk[["PLAYER_ID", "SEASON_YEAR", "FG3M", "ALPHA", "BETA"]]
    df = pairs.merge(season_t, on=["PLAYER_ID", "SEASON_YEAR"], how="left", validate="1:1")

    ml = preds[["PLAYER_ID", "START", "PRED_RIDGE", "PRED_GBM"]]
    df = df.merge(ml, on=["PLAYER_ID", "START"], how="left", validate="1:1")
    df["IN_ML_WINDOW"] = df["PRED_RIDGE"].notna().astype(int)

    df = flag_anomalies(df)
    df["PI_LOW_PCT"] = df["PI_LOW"] / df["FG3A_NEXT"]
    df["PI_HIGH_PCT"] = df["PI_HIGH"] / df["FG3A_NEXT"]
    df["VOLUME_BUCKET"] = pd.cut(df["FG3A"], BUCKET_EDGES, labels=BUCKET_LABELS).astype(str)

    df = df.sort_values(["START", "PLAYER_ID"]).reset_index(drop=True)
    df.insert(0, "PAIR_ID", np.arange(1, len(df) + 1))
    return df.rename(columns={"TARGET": "ACTUAL"})


def build_fact_forecast(fact_pair):
    """Long format: one row per pair x method, so one DAX measure scores any slice."""
    keys = ["PAIR_ID", "PLAYER_ID", "SEASON_YEAR", "VOLUME_BUCKET", "IN_ML_WINDOW"]
    pieces = []
    for key, (col, *_) in METHODS.items():
        part = fact_pair.loc[fact_pair[col].notna(), keys + ["ACTUAL", "FG3A_NEXT"]].copy()
        part["METHOD_KEY"] = key
        part["PREDICTION"] = fact_pair.loc[part.index, col]
        pieces.append(part)

    df = pd.concat(pieces, ignore_index=True).rename(columns={"FG3A_NEXT": "WEIGHT"})
    df["ERROR"] = df["PREDICTION"] - df["ACTUAL"]
    df["ABS_ERROR"] = df["ERROR"].abs()
    df["SQ_ERROR"] = df["ERROR"] ** 2
    return df


def build_dims(shrunk):
    """dim_player, dim_season, dim_method, dim_volume."""
    start = shrunk["SEASON_YEAR"].str[:4].astype(int)

    # A few players' names change spelling across seasons; keep the latest one.
    dim_player = (shrunk.assign(START=start)
                  .sort_values("START")
                  .groupby("PLAYER_ID", as_index=False)["PLAYER_NAME"].last())

    dim_season = (shrunk.assign(START=start)
                  .groupby(["SEASON_YEAR", "START"], as_index=False)
                  .agg(ALPHA=("ALPHA", "first"), BETA=("BETA", "first"),
                       PRIOR_MEAN=("PRIOR_MEAN", "first"), N_PLAYERS=("PLAYER_ID", "size")))
    dim_season["STRENGTH"] = dim_season["ALPHA"] + dim_season["BETA"]

    dim_method = pd.DataFrame(
        [(k, label, fam, order) for k, (_, label, fam, order) in METHODS.items()],
        columns=["METHOD_KEY", "METHOD_LABEL", "FAMILY", "SORT_ORDER"],
    )

    dim_volume = pd.DataFrame({
        "VOLUME_BUCKET": BUCKET_LABELS,
        "SORT_ORDER": range(1, len(BUCKET_LABELS) + 1),
        "MIN_3PA": BUCKET_EDGES[:-1],
    })
    return dim_player, dim_season, dim_method, dim_volume


FACT_KEYS = {  # fact column -> (dimension table, dimension key)
    "PLAYER_ID": ("dim_player", "PLAYER_ID"),
    "SEASON_YEAR": ("dim_season", "SEASON_YEAR"),
    "VOLUME_BUCKET": ("dim_volume", "VOLUME_BUCKET"),
    "METHOD_KEY": ("dim_method", "METHOD_KEY"),
}


def check_keys(tables):
    """Every fact key must exist in its dimension, and dimension keys must be unique."""
    for fact in ("fact_pair", "fact_forecast"):
        for col, (dim, key) in FACT_KEYS.items():
            if col not in tables[fact]:
                continue
            if tables[dim][key].duplicated().any():
                raise ValueError(f"{dim}.{key} is not unique")
            orphans = ~tables[fact][col].isin(tables[dim][key])
            if orphans.any():
                raise ValueError(f"{fact}.{col}: {orphans.sum()} values missing from {dim}")


def write_tables(tables, out_dir=OUT_DIR):
    """Check keys, then write one CSV per table."""
    check_keys(tables)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"  {name:<14} {len(df):>6,} rows")


def rmse(df):
    return np.sqrt(df["SQ_ERROR"].mean())


def summarize(fact_pair, fact_forecast):
    """Recompute the README's headline numbers from the new tables, as a check."""
    print("\nRMSE, all pairs (README: raw .0507, EB .0414, league .0438):")
    for key in ["raw", "eb", "league"]:
        print(f"  {key:<7} {rmse(fact_forecast[fact_forecast['METHOD_KEY'] == key]):.4f}")

    window = fact_forecast[fact_forecast["IN_ML_WINDOW"] == 1]
    print("\nRMSE, ML test window (README: EB .0417, ridge .0409, gbm .0409):")
    for key in METHODS:
        print(f"  {key:<7} {rmse(window[window['METHOD_KEY'] == key]):.4f}")

    flagged = fact_pair["IS_ANOMALY"]
    print(f"\nanomalies: {flagged.sum()} of {len(fact_pair):,} pairs ({flagged.mean():.1%})")
    print(fact_pair["DIRECTION"].value_counts().to_string())


def main():
    pairs, preds, shrunk = load_inputs()
    fact_pair = build_fact_pair(pairs, preds, shrunk)
    fact_forecast = build_fact_forecast(fact_pair)
    dim_player, dim_season, dim_method, dim_volume = build_dims(shrunk)

    print(f"\nwriting -> {OUT_DIR}")
    write_tables({
        "fact_pair": fact_pair, "fact_forecast": fact_forecast,
        "dim_player": dim_player, "dim_season": dim_season,
        "dim_method": dim_method, "dim_volume": dim_volume,
    })
    summarize(fact_pair, fact_forecast)


if __name__ == "__main__":
    main()

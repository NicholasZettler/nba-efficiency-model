# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Gold: validate shrinkage on next season
# MAGIC
# MAGIC **Replaces:** `04_validate.py` (same pairs, metrics, and leak-free baseline)
# MAGIC **Reads:** `nba_gold_shrunk`
# MAGIC **Writes:** `nba_gold_validation_pairs`, `nba_gold_validation_metrics`

# COMMAND ----------

import numpy as np
import pandas as pd
from scipy.stats import linregress

SCHEMA = "workspace.default"


def load_table(name):
    return spark.table(f"{SCHEMA}.{name}").toPandas()


def save_table(pdf, name):
    (spark.createDataFrame(pdf)
          .write.mode("overwrite")
          .option("overwriteSchema", "true")
          .saveAsTable(f"{SCHEMA}.{name}"))
    print(f"wrote {len(pdf):,} rows -> {SCHEMA}.{name}")


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
        left = df[df["SEASON_YEAR"] == a][["PLAYER_ID", "PLAYER_NAME", "FG3A", "FG3_PCT_RAW",
                                           "FG3_PCT_SHRUNK", "PRIOR_MEAN"]]
        right = df[df["SEASON_YEAR"] == b][["PLAYER_ID", "FG3A", "FG3_PCT_RAW"]]
        merged = left.merge(right, on="PLAYER_ID", suffixes=("_T", "_NEXT"))
        merged["PAIR"] = f"{a} -> {b}"
        pairs.append(merged)
    return pd.concat(pairs, ignore_index=True)


def score(pred, actual, weights):
    """RMSE, weighted RMSE, MAE, and calibration slope / correlation."""
    err = pred - actual
    out = {"RMSE": np.sqrt(np.mean(err ** 2)),
           "wRMSE": np.sqrt(np.sum(weights * err ** 2) / np.sum(weights)),
           "MAE": np.mean(np.abs(err)), "slope": np.nan, "r": np.nan}
    if pred.max() > pred.min():  # a constant prediction has no slope to fit
        fit = linregress(pred, actual)
        out.update(slope=fit.slope, r=fit.rvalue)
    return out


def evaluate(df, group_type, group):
    """One metrics row per method, plus the RMSE gain from shrinkage."""
    actual = df["FG3_PCT_RAW_NEXT"].to_numpy()
    w = df["FG3A_NEXT"].to_numpy(dtype=float)
    methods = {
        "raw": df["FG3_PCT_RAW_T"].to_numpy(),
        "shrunk": df["FG3_PCT_SHRUNK"].to_numpy(),
        # Baseline: season t's prior mean for everyone. Known at prediction time.
        "baseline": df["PRIOR_MEAN"].to_numpy(),
    }
    rows = [{"GROUP_TYPE": group_type, "GROUP": group, "N": len(df), "METHOD": m,
             **score(pred, actual, w)} for m, pred in methods.items()]
    gain = 100 * (rows[0]["RMSE"] - rows[1]["RMSE"]) / rows[0]["RMSE"]
    print(f"{group:>22}  n={len(df):>5,}  RMSE gain from shrinkage: {gain:+.2f}%")
    return rows

# COMMAND ----------

df = load_table("nba_gold_shrunk")
pairs = build_pairs(df)
print(f"built {len(pairs):,} player-season pairs\n")

rows = evaluate(pairs, "pooled", "ALL PAIRS POOLED")
for name, grp in pairs.groupby("PAIR"):
    rows += evaluate(grp, "season_pair", name)

pairs["BUCKET"] = pd.cut(pairs["FG3A_T"], bins=[0, 100, 200, 400, np.inf],
                         labels=["<100 3PA", "100-199", "200-399", "400+"])
for name, grp in pairs.groupby("BUCKET", observed=True):
    if len(grp) >= 20:
        rows += evaluate(grp, "volume_bucket", name)

pairs["BUCKET"] = pairs["BUCKET"].astype(str)
metrics = pd.DataFrame(rows)

# COMMAND ----------

save_table(pairs, "nba_gold_validation_pairs")
save_table(metrics, "nba_gold_validation_metrics")
display(metrics[metrics["GROUP_TYPE"] != "season_pair"])

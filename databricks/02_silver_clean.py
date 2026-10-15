# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver: game logs → player-seasons
# MAGIC
# MAGIC **Replaces:** `02_clean.py` (same logic and thresholds)
# MAGIC **Reads:** `nba_bronze_gamelogs`, `nba_bronze_ages`
# MAGIC **Writes:** `nba_silver_player_seasons` (qualified player-seasons only)

# COMMAND ----------

import numpy as np
import pandas as pd

SCHEMA = "workspace.default"
MIN_3PA_PER_GAME = 1.5
MIN_GAMES = 20


def load_table(name):
    """Read a Delta table into pandas, dropping bronze lineage columns."""
    pdf = spark.table(f"{SCHEMA}.{name}").toPandas()
    return pdf.drop(columns=[c for c in pdf.columns if c.startswith("_")])


def save_table(pdf, name):
    (spark.createDataFrame(pdf)
          .write.mode("overwrite")
          .option("overwriteSchema", "true")
          .saveAsTable(f"{SCHEMA}.{name}"))
    print(f"wrote {len(pdf):,} rows -> {SCHEMA}.{name}")


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
    total = out["FG3A"].sum()
    rows = []
    for cut in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        sub = out[(out["FG3A_PG"] >= cut) & (out["G"] >= MIN_GAMES)]
        rows.append({"min_3PA_per_game": cut, "players": len(sub),
                     "median_3PA": sub["FG3A"].median(), "min_3PA": sub["FG3A"].min(),
                     "pct_attempts_kept": round(100 * sub["FG3A"].sum() / total, 1)})
    return pd.DataFrame(rows)

# COMMAND ----------

raw = load_table("nba_bronze_gamelogs")
print(f"loaded {raw['SEASON_YEAR'].nunique()} seasons -> {len(raw):,} game rows")

out = collapse(raw)
ages = load_table("nba_bronze_ages").drop_duplicates(subset=["SEASON_YEAR", "PLAYER_ID"])
out = out.merge(ages, on=["SEASON_YEAR", "PLAYER_ID"], how="left")
print(f"collapsed to {len(out):,} player-seasons ({out['AGE'].isna().sum()} missing AGE)")

display(sweep_thresholds(out))

# COMMAND ----------

qualified = out[(out["FG3A_PG"] >= MIN_3PA_PER_GAME) & (out["G"] >= MIN_GAMES)].copy()
kept = 100 * qualified["FG3A"].sum() / out["FG3A"].sum()
print(f"qualified: {len(qualified):,} player-seasons, keeping {kept:.1f}% of all 3PA")

save_table(qualified, "nba_silver_player_seasons")

latest = qualified["SEASON_YEAR"].max()
display(qualified[qualified["SEASON_YEAR"] == latest]
        .nlargest(5, "FG3A")[["PLAYER_NAME", "AGE", "FG3M", "FG3A",
                              "FG3_PCT_RAW", "FT_PCT", "FG3A_RATE"]])

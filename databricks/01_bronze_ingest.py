# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze: ingest raw CSVs into Delta
# MAGIC
# MAGIC **Replaces:** nothing (new step). `01_pull.py` still runs locally, because Free Edition limits outbound internet.
# MAGIC
# MAGIC **Reads:** the 20 CSVs uploaded to the `nba_raw` volume
# MAGIC **Writes:** `nba_bronze_gamelogs`, `nba_bronze_ages`
# MAGIC
# MAGIC Bronze = raw data, unchanged, plus two lineage columns (`_SOURCE_FILE`, `_INGESTED_AT`) so every row can be traced back to the file it came from. Data-quality checks stop the pipeline if the raw data is incomplete or impossible.

# COMMAND ----------

import glob
import os

import pandas as pd

VOLUME = "/Volumes/workspace/default/nba_raw"
SCHEMA = "workspace.default"
EXPECTED_SEASONS = 10


def save_table(pdf, name):
    """Write a pandas DataFrame as a managed Delta table (replacing any old version)."""
    (spark.createDataFrame(pdf)
          .write.mode("overwrite")
          .option("overwriteSchema", "true")
          .saveAsTable(f"{SCHEMA}.{name}"))
    print(f"wrote {len(pdf):,} rows -> {SCHEMA}.{name}")


def read_folder(pattern):
    """Stack every CSV matching pattern, tagging each row with its source file."""
    files = sorted(glob.glob(pattern))
    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["_SOURCE_FILE"] = os.path.basename(path)
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No files match {pattern}")
    out = pd.concat(frames, ignore_index=True)
    out["_INGESTED_AT"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return out, len(files)

# COMMAND ----------

logs, n_log_files = read_folder(f"{VOLUME}/gamelogs_*.csv")
ages, n_age_files = read_folder(f"{VOLUME}/ages/ages_*.csv")
print(f"game logs: {n_log_files} files, {len(logs):,} rows")
print(f"ages:      {n_age_files} files, {len(ages):,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data-quality checks
# MAGIC Hard checks fail the run. Soft checks are reported so you know what `02` will clean up.

# COMMAND ----------

hard_checks = {
    f"{EXPECTED_SEASONS} game-log files": n_log_files == EXPECTED_SEASONS,
    f"{EXPECTED_SEASONS} age files": n_age_files == EXPECTED_SEASONS,
    "no missing PLAYER_ID / GAME_ID": logs[["PLAYER_ID", "GAME_ID"]].notna().all().all(),
    "no negative shot counts": (logs[["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA"]] >= 0).all().all(),
    "makes never exceed attempts": ((logs["FG3M"] <= logs["FG3A"])
                                    & (logs["FGM"] <= logs["FGA"])
                                    & (logs["FTM"] <= logs["FTA"])).all(),
    "every season has game rows": logs.groupby("SEASON_YEAR").size().min() > 0,
}
soft_checks = {
    "duplicate (PLAYER_ID, GAME_ID) rows": int(logs.duplicated(["PLAYER_ID", "GAME_ID"]).sum()),
    "duplicate (SEASON_YEAR, PLAYER_ID) age rows": int(ages.duplicated(["SEASON_YEAR", "PLAYER_ID"]).sum()),
    "age rows missing AGE": int(ages["AGE"].isna().sum()),
}

for name, ok in hard_checks.items():
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
for name, count in soft_checks.items():
    print(f"INFO  {name}: {count}")

failed = [name for name, ok in hard_checks.items() if not ok]
if failed:
    raise ValueError(f"Data-quality checks failed: {failed}")

# COMMAND ----------

save_table(logs, "nba_bronze_gamelogs")
save_table(ages, "nba_bronze_ages")

display(logs.groupby("SEASON_YEAR").size().rename("game_rows").reset_index())

"""Pull NBA game logs from the stats API and cache them as raw CSVs."""

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import playergamelogs

SEASONS = ["2018-19", "2019-20", "2021-22", "2022-23", "2023-24", "2024-25"]

KEEP_COLS = [
    "SEASON_YEAR", "PLAYER_ID", "PLAYER_NAME",
    "GAME_ID", "GAME_DATE", "MIN", "FG3M", "FG3A",
]

RAW_DIR = Path("data/raw")


def fetch_season(season, retries=3):
    """Fetch one season of game logs, retrying on timeout."""
    for attempt in range(retries):
        try:
            logs = playergamelogs.PlayerGameLogs(
                season_nullable=season,
                season_type_nullable="Regular Season",
            )
            return logs.get_data_frames()[0]
        except Exception as e:
            print(f"  attempt {attempt + 1} failed: {e}")
            time.sleep(5)
    raise RuntimeError(f"Could not fetch {season}")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for season in SEASONS:
        out_path = RAW_DIR / f"gamelogs_{season}.csv"

        if out_path.exists():
            print(f"{season}: cached, skipping")
            continue

        print(f"{season}: fetching...")
        df = fetch_season(season)
        df = df[KEEP_COLS]
        df.to_csv(out_path, index=False)
        print(f"{season}: saved {len(df):,} rows")

        time.sleep(1)  # be polite to the API

    print("\nDone.")


if __name__ == "__main__":
    main()

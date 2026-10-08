"""Pull NBA game logs and player ages from the stats API and cache them as raw CSVs."""

import time
from pathlib import Path

from nba_api.stats.endpoints import leaguedashplayerstats, playergamelogs

SEASONS = [
    "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
]

KEEP_COLS = [
    "SEASON_YEAR", "PLAYER_ID", "PLAYER_NAME",
    "GAME_ID", "GAME_DATE", "MIN",
    "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
]

RAW_DIR = Path("data/raw_v2")
AGE_DIR = RAW_DIR / "ages"


def with_retries(fetch, label, retries=3):
    """Call fetch() and return its first data frame, retrying on failure."""
    for attempt in range(retries):
        try:
            return fetch().get_data_frames()[0]
        except Exception as e:
            print(f"  {label}: attempt {attempt + 1} failed: {e}")
            time.sleep(5)
    raise RuntimeError(f"Could not fetch {label}")


def fetch_gamelogs(season):
    return with_retries(
        lambda: playergamelogs.PlayerGameLogs(
            season_nullable=season,
            season_type_nullable="Regular Season",
        ),
        f"game logs {season}",
    )


def fetch_ages(season):
    """One row per player with AGE for the season (one API call)."""
    df = with_retries(
        lambda: leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
        ),
        f"ages {season}",
    )
    df = df[["PLAYER_ID", "AGE"]].copy()
    df.insert(0, "SEASON_YEAR", season)
    return df


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    AGE_DIR.mkdir(parents=True, exist_ok=True)

    for season in SEASONS:
        logs_path = RAW_DIR / f"gamelogs_{season}.csv"
        if logs_path.exists():
            print(f"{season}: game logs cached, skipping")
        else:
            print(f"{season}: fetching game logs...")
            df = fetch_gamelogs(season)[KEEP_COLS]
            df.to_csv(logs_path, index=False)
            print(f"{season}: saved {len(df):,} game rows")
            time.sleep(1)  # be polite to the API

        age_path = AGE_DIR / f"ages_{season}.csv"
        if age_path.exists():
            print(f"{season}: ages cached, skipping")
        else:
            print(f"{season}: fetching ages...")
            ages = fetch_ages(season)
            ages.to_csv(age_path, index=False)
            print(f"{season}: saved {len(ages):,} player ages")
            time.sleep(1)

    print("\nDone.")


if __name__ == "__main__":
    main()

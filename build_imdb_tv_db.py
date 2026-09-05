#!/usr/bin/env python3
"""
IMDb TV Shows & Seasons Relational Database Builder
- Downloads basics, ratings, and episode dumps from IMDb
- Aggregates episode metrics into Season-level data (avg rating, total votes, ep count)
- Treats unknown/unassigned seasons as 'Specials' (Season 0)
- Normalizes into a clean 1-to-Many SQLite schema (tv_shows -> tv_seasons)
"""

import logging
import sqlite3
import urllib.request
from datetime import datetime
from pathlib import Path
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("imdb_tv_seasons")

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

SQLITE_DB = DATA_DIR / "imdb_tv_shows.db"

IMDB_BASICS_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
IMDB_RATINGS_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"
IMDB_EPISODES_URL = "https://datasets.imdbws.com/title.episode.tsv.gz"


def download_file(url: str, dest: Path):
    logger.info(f"📥 Downloading {dest.name}...")
    urllib.request.urlretrieve(url, dest)
    logger.info(f"✅ Downloaded {dest.name}")


def main():
    basics_gz = DATA_DIR / "title.basics.tsv.gz"
    ratings_gz = DATA_DIR / "title.ratings.tsv.gz"
    episodes_gz = DATA_DIR / "title.episode.tsv.gz"

    # 1. Download the 3 required IMDb dumps (~250MB total)
    download_file(IMDB_BASICS_URL, basics_gz)
    download_file(IMDB_RATINGS_URL, ratings_gz)
    download_file(IMDB_EPISODES_URL, episodes_gz)

    # 2. Extract TV Series from title.basics
    logger.info("Extracting TV shows from title.basics...")
    tv_chunks = []
    for chunk in pd.read_csv(
        basics_gz,
        sep="\t",
        na_values=r"\N",
        usecols=[
            "tconst",
            "titleType",
            "primaryTitle",
            "originalTitle",
            "isAdult",
            "startYear",
            "endYear",
            "runtimeMinutes",
            "genres",
        ],
        dtype=str,
        chunksize=250_000,
    ):
        tv_filter = chunk["titleType"].isin(["tvSeries", "tvMiniSeries"])
        tv_chunks.append(chunk[tv_filter])

    df_shows = pd.concat(tv_chunks, ignore_index=True)
    basics_gz.unlink(missing_ok=True)
    logger.info(f"Loaded {len(df_shows):,} total TV shows.")

    # 3. Load IMDb Ratings
    logger.info("Loading ratings...")
    df_ratings = pd.read_csv(
        ratings_gz,
        sep="\t",
        na_values=r"\N",
        usecols=["tconst", "averageRating", "numVotes"],
        dtype={"tconst": str, "averageRating": float, "numVotes": float},
    )
    ratings_gz.unlink(missing_ok=True)

    # Merge overall ratings into shows
    df_shows = df_shows.merge(df_ratings, on="tconst", how="left")
    df_shows.rename(
        columns={
            "tconst": "imdb_id",
            "primaryTitle": "title",
            "originalTitle": "original_title",
            "titleType": "type",
            "startYear": "start_year",
            "endYear": "end_year",
            "runtimeMinutes": "runtime_minutes",
            "averageRating": "imdb_rating",
            "numVotes": "imdb_votes",
            "isAdult": "is_adult",
        },
        inplace=True,
    )

    df_shows["start_year"] = pd.to_numeric(df_shows["start_year"], errors="coerce")
    df_shows["end_year"] = pd.to_numeric(df_shows["end_year"], errors="coerce")
    df_shows["runtime_minutes"] = pd.to_numeric(
        df_shows["runtime_minutes"], errors="coerce"
    )
    df_shows["imdb_votes"] = (
        pd.to_numeric(df_shows["imdb_votes"], errors="coerce").fillna(0).astype(int)
    )

    # 4. Filter Shows: Popular shows (>= 50 votes) OR recent/upcoming (2024–present / NaN)
    current_year = datetime.now().year
    logger.info("Filtering parent TV shows...")
    keep_condition = (
        (df_shows["imdb_votes"] >= 50)
        | (df_shows["start_year"] >= (current_year - 1))
        | (df_shows["start_year"].isna())
    )
    shows_table = df_shows[keep_condition].copy()
    valid_series_ids = set(shows_table["imdb_id"])
    logger.info(f"Curated {len(shows_table):,} shows.")

    # 5. Process Episodes and Aggregate into Seasons
    logger.info("Parsing episodes and computing season-level metrics...")
    df_episodes = pd.read_csv(
        episodes_gz,
        sep="\t",
        na_values=r"\N",
        usecols=["tconst", "parentTconst", "seasonNumber", "episodeNumber"],
        dtype={
            "tconst": str,
            "parentTconst": str,
            "seasonNumber": str,
            "episodeNumber": str,
        },
    )
    episodes_gz.unlink(missing_ok=True)

    # Keep only episodes belonging to our curated shows (huge speedup!)
    df_episodes = df_episodes[df_episodes["parentTconst"].isin(valid_series_ids)]

    # Attach individual episode ratings & votes
    df_episodes = df_episodes.merge(df_ratings, on="tconst", how="left")

    # Clean seasonNumber: treat missing/unknown as Season 0 ("Specials")
    df_episodes["season_num"] = (
        pd.to_numeric(df_episodes["seasonNumber"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # Aggregate by Show + Season
    logger.info("Aggregating seasons...")
    season_agg = (
        df_episodes.groupby(["parentTconst", "season_num"])
        .agg(
            episode_count=("tconst", "count"),
            rated_episodes=("averageRating", "count"),
            avg_rating=("averageRating", "mean"),
            total_votes=("numVotes", "sum"),
        )
        .reset_index()
    )

    season_agg.rename(
        columns={"parentTconst": "series_id", "season_num": "season_number"},
        inplace=True,
    )
    season_agg["avg_rating"] = season_agg["avg_rating"].round(1)
    season_agg["total_votes"] = season_agg["total_votes"].fillna(0).astype(int)

    # Season display name: Season 0 -> 'Specials', Season 1 -> 'Season 1'
    season_agg["season_name"] = season_agg["season_number"].apply(
        lambda s: "Specials" if s == 0 else f"Season {s}"
    )

    # Sort seasons logically
    season_agg.sort_values(by=["series_id", "season_number"], inplace=True)

    # 6. Save to SQLite with Normalized 1-to-Many Schema
    logger.info(f"Writing normalized database to SQLite ({SQLITE_DB})...")
    if SQLITE_DB.exists():
        SQLITE_DB.unlink()

    conn = sqlite3.connect(SQLITE_DB)
    cur = conn.cursor()

    # Enable Foreign Key support in SQLite
    cur.execute("PRAGMA foreign_keys = ON;")

    # Create parent table: tv_shows
    cur.execute("""
        CREATE TABLE tv_shows (
            imdb_id TEXT PRIMARY KEY,
            title TEXT,
            original_title TEXT,
            type TEXT,
            start_year INTEGER,
            end_year INTEGER,
            runtime_minutes INTEGER,
            genres TEXT,
            is_adult TEXT,
            imdb_rating REAL,
            imdb_votes INTEGER
        );
    """)

    # Create child table: tv_seasons (1-to-Many)
    cur.execute("""
        CREATE TABLE tv_seasons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            series_id TEXT NOT NULL,
            season_number INTEGER NOT NULL,
            season_name TEXT,
            episode_count INTEGER,
            rated_episodes INTEGER,
            avg_rating REAL,
            total_votes INTEGER,
            FOREIGN KEY (series_id) REFERENCES tv_shows(imdb_id) ON DELETE CASCADE
        );
    """)

    # Insert Data
    shows_table.to_sql("tv_shows", conn, if_exists="append", index=False)
    season_agg.to_sql("tv_seasons", conn, if_exists="append", index=False)

    # Create Indexes for lightning-fast queries in DB Browser
    logger.info("Creating indexes...")
    cur.execute("CREATE INDEX idx_shows_title ON tv_shows(title);")
    cur.execute("CREATE INDEX idx_shows_rating ON tv_shows(imdb_rating);")
    cur.execute("CREATE INDEX idx_seasons_series_id ON tv_seasons(series_id);")
    cur.execute("CREATE INDEX idx_seasons_rating ON tv_seasons(avg_rating);")
    cur.execute("CREATE INDEX idx_seasons_num ON tv_seasons(season_number);")

    conn.commit()
    conn.close()

    logger.info(
        f"🎉 All done! Saved {len(shows_table):,} shows and {len(season_agg):,} seasons to '{SQLITE_DB}'."
    )


if __name__ == "__main__":
    main()

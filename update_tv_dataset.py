#!/usr/bin/env python3
"""
TMDb + IMDb TV Shows Master Pipeline
- Bootstraps from base CSV
- Scrapes newly released shows via TMDb API with caching
- Downloads official IMDb dumps (basics + ratings)
- Maps IMDb IDs, ratings, and vote counts to BOTH historical and new shows
- Exports cleanly to a SQLite database for DB Browser for SQLite
"""

import concurrent.futures
import gzip
import json
import logging
import os
import sqlite3
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("tv_updater")

load_dotenv()
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
if not TMDB_API_KEY:
    logger.error(
        "❌ TMDB_API_KEY environment variable required in .env or environment!"
    )
    sys.exit(1)

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_CSV = DATA_DIR / "TMDB_tv_dataset_v3.csv"
CACHE_JSON = DATA_DIR / "scraped_new_shows_cache.json"
SQLITE_DB = DATA_DIR / "tv_shows.db"
OUTPUT_CSV = DATA_DIR / "TMDB_all_tv_shows.csv"

IMDB_RATINGS_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"
IMDB_BASICS_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"


# ==========================================
# 1. TMDb Daily ID Export
# ==========================================
def get_daily_tmdb_tv_ids():
    today = datetime.now(timezone.utc)
    date_str = today.strftime("%m_%d_%Y")
    url = f"http://files.tmdb.org/p/exports/tv_series_ids_{date_str}.json.gz"
    local_gz = DATA_DIR / f"tv_ids_{date_str}.json.gz"

    logger.info(f"Downloading daily TMDb TV export: {url}...")
    try:
        urllib.request.urlretrieve(url, local_gz)
    except Exception:
        yesterday = today - pd.Timedelta(days=1)
        date_str = yesterday.strftime("%m_%d_%Y")
        url = f"http://files.tmdb.org/p/exports/tv_series_ids_{date_str}.json.gz"
        local_gz = DATA_DIR / f"tv_ids_{date_str}.json.gz"
        urllib.request.urlretrieve(url, local_gz)

    tv_ids = set()
    with gzip.open(local_gz, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    tv_ids.add(int(json.loads(line)["id"]))
                except Exception:
                    continue

    local_gz.unlink(missing_ok=True)
    logger.info(f"✅ Found {len(tv_ids):,} total TV shows currently on TMDb.")
    return tv_ids


# ==========================================
# 2. TMDb API Show Details Scraper
# ==========================================
def fetch_tv_show(tv_id):
    url = (
        f"https://api.themoviedb.org/3/tv/{tv_id}"
        f"?api_key={TMDB_API_KEY}"
        f"&append_to_response=external_ids,credits"
    )
    try:
        res = requests.get(url, timeout=15)
        if res.status_code == 429:
            retry = int(res.headers.get("Retry-After", 5))
            time.sleep(retry)
            return fetch_tv_show(tv_id)
        if res.status_code != 200:
            return None

        d = res.json()
        ext = d.get("external_ids") or {}
        imdb_id = ext.get("imdb_id")

        genres = ", ".join([g["name"] for g in d.get("genres", [])])
        networks = ", ".join([n["name"] for n in d.get("networks", [])])
        companies = ", ".join([c["name"] for c in d.get("production_companies", [])])
        showrunners = ", ".join([c["name"] for c in d.get("created_by", [])])

        credits_data = d.get("credits") or {}
        cast = ", ".join([c["name"] for c in credits_data.get("cast", [])[:12]])

        return {
            "id": d.get("id"),
            "name": d.get("name"),
            "number_of_seasons": d.get("number_of_seasons"),
            "number_of_episodes": d.get("number_of_episodes"),
            "original_language": d.get("original_language"),
            "vote_count": d.get("vote_count"),
            "vote_average": d.get("vote_average"),
            "overview": d.get("overview"),
            "adult": d.get("adult"),
            "backdrop_path": d.get("backdrop_path"),
            "poster_path": d.get("poster_path"),
            "first_air_date": d.get("first_air_date"),
            "last_air_date": d.get("last_air_date"),
            "status": d.get("status"),
            "genres": genres,
            "networks": networks,
            "production_companies": companies,
            "created_by": showrunners,
            "cast": cast,
            "imdb_id": imdb_id,
        }
    except Exception:
        return None


# ==========================================
# 3. IMDb Dumps (Basics + Ratings) Merger
# ==========================================
def download_and_merge_imdb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downloads IMDb basics and ratings dumps, maps them to both:
    1. Shows that already have imdb_id (exact ID join)
    2. Historical shows missing imdb_id (fuzzy title + year match)
    """
    ratings_gz = DATA_DIR / "title.ratings.tsv.gz"
    basics_gz = DATA_DIR / "title.basics.tsv.gz"

    logger.info("📥 Downloading IMDb title.ratings.tsv.gz...")
    urllib.request.urlretrieve(IMDB_RATINGS_URL, ratings_gz)

    logger.info("📥 Downloading IMDb title.basics.tsv.gz...")
    urllib.request.urlretrieve(IMDB_BASICS_URL, basics_gz)

    # 1. Load Ratings
    logger.info("Processing IMDb ratings...")
    ratings = pd.read_csv(
        ratings_gz,
        sep="\t",
        na_values=r"\N",
        usecols=["tconst", "averageRating", "numVotes"],
        dtype={"tconst": str, "averageRating": float, "numVotes": float},
    ).rename(
        columns={
            "tconst": "imdb_id",
            "averageRating": "imdb_rating",
            "numVotes": "imdb_votes",
        }
    )

    # 2. Load Basics (Filter only TV series to save memory)
    logger.info("Processing IMDb TV series titles & years...")
    basics = pd.read_csv(
        basics_gz,
        sep="\t",
        na_values=r"\N",
        usecols=["tconst", "titleType", "primaryTitle", "startYear"],
        dtype=str,
    )
    basics = basics[basics["titleType"].isin(["tvSeries", "tvMiniSeries"])]
    basics.rename(columns={"tconst": "imdb_id"}, inplace=True)

    # Combine basics + ratings for title-matching
    imdb_combined = basics.merge(ratings, on="imdb_id", how="left")
    imdb_combined["clean_title"] = (
        imdb_combined["primaryTitle"].astype(str).str.strip().str.lower()
    )
    imdb_combined["startYear"] = pd.to_numeric(
        imdb_combined["startYear"], errors="coerce"
    )

    # If there are duplicates in title + year, keep the one with the most votes
    imdb_combined.sort_values(by="imdb_votes", ascending=False, inplace=True)
    imdb_lookup = imdb_combined.drop_duplicates(
        subset=["clean_title", "startYear"], keep="first"
    )

    # Clean up downloaded gz files immediately
    ratings_gz.unlink(missing_ok=True)
    basics_gz.unlink(missing_ok=True)

    # Ensure imdb_id exists
    if "imdb_id" not in df.columns:
        df["imdb_id"] = None
    df["imdb_id"] = (
        df["imdb_id"].astype(str).replace({"nan": None, "None": None, "": None})
    )

    # First Pass: Direct merge for shows that ALREADY have imdb_id
    logger.info("Merging IMDb stats on existing imdb_id...")
    df.drop(columns=["imdb_rating", "imdb_votes"], errors="ignore", inplace=True)
    df = df.merge(ratings, on="imdb_id", how="left")

    # Second Pass: Fill missing imdb_id, imdb_rating, imdb_votes using title + air year
    missing_mask = df["imdb_id"].isna() | df["imdb_rating"].isna()
    logger.info(
        f"Attempting title+year match for {missing_mask.sum():,} shows missing IMDb stats..."
    )

    df["air_year"] = pd.to_datetime(df["first_air_date"], errors="coerce").dt.year
    df["clean_name"] = df["name"].astype(str).str.strip().str.lower()

    # Merge on clean_name + air_year
    title_matches = df[missing_mask][["id", "clean_name", "air_year"]].merge(
        imdb_lookup[
            ["clean_title", "startYear", "imdb_id", "imdb_rating", "imdb_votes"]
        ],
        left_on=["clean_name", "air_year"],
        right_on=["clean_title", "startYear"],
        how="inner",
    )

    if not title_matches.empty:
        title_matches.drop_duplicates(subset=["id"], inplace=True)
        title_matches.set_index("id", inplace=True)

        # Update the main dataframe
        for col in ["imdb_id", "imdb_rating", "imdb_votes"]:
            df.loc[df["id"].isin(title_matches.index) & df[col].isna(), col] = df[
                "id"
            ].map(title_matches[col])

    # Clean temporary helper columns
    df.drop(columns=["air_year", "clean_name"], errors="ignore", inplace=True)
    logger.info("✅ IMDb matching complete.")
    return df


# ==========================================
# 4. Save to SQLite Database
# ==========================================
def save_to_sqlite(df: pd.DataFrame, db_path: Path):
    logger.info(f"Creating SQLite database at {db_path}...")
    # Remove existing DB file if replacing
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    # Write DataFrame to SQL
    df.to_sql("tv_shows", conn, if_exists="replace", index=False)

    # Create helpful indexes so queries inside DB Browser are super fast
    logger.info("Creating SQLite indexes (id, name, imdb_rating, vote_count)...")
    cur = conn.cursor()
    cur.execute("CREATE INDEX idx_tv_id ON tv_shows(id);")
    cur.execute("CREATE INDEX idx_tv_name ON tv_shows(name);")
    cur.execute("CREATE INDEX idx_tv_imdb_id ON tv_shows(imdb_id);")
    cur.execute("CREATE INDEX idx_tv_imdb_rating ON tv_shows(imdb_rating);")
    cur.execute("CREATE INDEX idx_tv_vote_count ON tv_shows(vote_count);")
    conn.commit()
    conn.close()
    logger.info(f"✅ SQLite database saved to {db_path}!")


# ==========================================
# 5. Main Runner
# ==========================================
def main():
    if not BASE_CSV.exists():
        logger.error(
            f"❌ Base CSV {BASE_CSV} not found! Put Asaniczka's CSV in data/ directory."
        )
        return

    logger.info(f"Reading base dataset ({BASE_CSV})...")
    base_df = pd.read_csv(BASE_CSV, low_memory=False)
    if "imdb_id" not in base_df.columns:
        base_df["imdb_id"] = None

    existing_ids = set(base_df["id"].dropna().astype(int))
    logger.info(f"Loaded {len(existing_ids):,} historical TV shows.")

    # Check if we already have cached scrape results from a prior run
    new_shows = []
    if CACHE_JSON.exists():
        logger.info(
            f"⚡ Found cached scrape results ({CACHE_JSON}), loading without scraping..."
        )
        with open(CACHE_JSON, "r", encoding="utf-8") as f:
            new_shows = json.load(f)
    else:
        daily_ids = get_daily_tmdb_tv_ids()
        new_ids = list(daily_ids - existing_ids)
        logger.info(f"💡 Found {len(new_ids):,} new TV shows to scrape.")

        if new_ids:
            logger.info(f"⚡ Scraping {len(new_ids):,} shows using 20 threads...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                futures = {
                    executor.submit(fetch_tv_show, s_id): s_id for s_id in new_ids
                }
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(new_ids),
                    unit="show",
                ):
                    res = future.result()
                    if res:
                        new_shows.append(res)

            # SAVE CACHE IMMEDIATELY TO PREVENT DATA LOSS!
            with open(CACHE_JSON, "w", encoding="utf-8") as f:
                json.dump(new_shows, f)
            logger.info(
                f"💾 Checkpointed {len(new_shows):,} scraped shows to {CACHE_JSON}"
            )

    if new_shows:
        new_df = pd.DataFrame(new_shows)
        combined_df = pd.concat([base_df, new_df], ignore_index=True)
        combined_df.drop_duplicates(subset=["id"], keep="last", inplace=True)
    else:
        combined_df = base_df

    # Merge IMDb basics + ratings
    final_df = download_and_merge_imdb(combined_df)

    # Filter out empty/unwatched junk shows
    logger.info(
        "Filtering for real/watched shows (vote_count >= 5 or imdb_votes >= 50)..."
    )
    filtered_df = final_df[
        (final_df["vote_count"].fillna(0) >= 5)
        | (final_df["imdb_votes"].fillna(0) >= 50)
    ].copy()

    # Save to CSV as backup
    filtered_df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"💾 Saved CSV to {OUTPUT_CSV}")

    # Save directly to SQLite DB for DB Browser
    save_to_sqlite(filtered_df, SQLITE_DB)

    # Clean up temp scrape cache on complete success
    CACHE_JSON.unlink(missing_ok=True)
    logger.info(
        "🎉 All done! You can now open 'data/tv_shows.db' in DB Browser for SQLite."
    )


if __name__ == "__main__":
    main()

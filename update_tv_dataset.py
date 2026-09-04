#!/usr/bin/env python3
"""
TMDb + IMDb TV Shows Updater
Bootstraps from Asaniczka's 150k base CSV, scrapes missing 2024-2026 TV shows,
and merges official IMDb ratings and vote counts.
"""

import concurrent.futures
import gzip
import json
import logging
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tv_updater")


load_dotenv()
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
if not TMDB_API_KEY:
    logger.error("❌ TMDB_API_KEY environment variable required!")
    sys.exit(1)

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_CSV = DATA_DIR / "TMDB_tv_dataset_v3.csv"
OUTPUT_CSV = DATA_DIR / "TMDB_all_tv_shows.csv"
IMDB_RATINGS_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"


# 1. Download Today's Complete List of TV Show IDs from TMDb
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
                tv_ids.add(json.loads(line)["id"])

    local_gz.unlink(missing_ok=True)
    logger.info(f"✅ Found {len(tv_ids):,} total TV shows currently on TMDb.")
    return tv_ids


# 2. Fetch Single TV Show Details from TMDb API
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
            "showrunners": showrunners,
            "cast": cast,
            "imdb_id": imdb_id,
        }
    except Exception:
        return None


# 3. Merge Official IMDb Ratings Dump
def merge_imdb_ratings(df):
    logger.info("Downloading official IMDb daily ratings dump...")
    imdb_gz = DATA_DIR / "title.ratings.tsv.gz"
    urllib.request.urlretrieve(IMDB_RATINGS_URL, imdb_gz)

    logger.info("Merging IMDb ratings on imdb_id...")
    ratings = pd.read_csv(
        imdb_gz,
        sep="\t",
        usecols=["tconst", "averageRating", "numVotes"],
        compression="gzip",
        dtype={"tconst": str, "averageRating": float, "numVotes": float},
    )
    ratings.rename(
        columns={"tconst": "imdb_id", "averageRating": "imdb_rating", "numVotes": "imdb_votes"},
        inplace=True,
    )

    df.drop(columns=["imdb_rating", "imdb_votes"], errors="ignore", inplace=True)
    merged = df.merge(ratings, on="imdb_id", how="left")
    imdb_gz.unlink(missing_ok=True)
    return merged


def main():
    if not BASE_CSV.exists():
        logger.error(f"❌ Base CSV {BASE_CSV} not found! Download it from Kaggle first.")
        return

    logger.info(f"Reading Asaniczka's base dataset ({BASE_CSV})...")
    base_df = pd.read_csv(BASE_CSV, low_memory=False)
    existing_ids = set(base_df["id"].dropna().astype(int))
    logger.info(f"Loaded {len(existing_ids):,} historical TV shows.")

    daily_ids = get_daily_tmdb_tv_ids()
    new_ids = list(daily_ids - existing_ids)
    logger.info(f"💡 Found {len(new_ids):,} NEW TV shows released between 2024 and today.")

    new_shows = []
    if new_ids:
        logger.info(f"⚡ Scraping {len(new_ids):,} shows using 20 parallel threads (~15 mins)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(fetch_tv_show, s_id): s_id for s_id in new_ids}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(new_ids), unit="show"):
                res = future.result()
                if res:
                    new_shows.append(res)

    if new_shows:
        new_df = pd.DataFrame(new_shows)
        combined_df = pd.concat([base_df, new_df], ignore_index=True)
        combined_df.drop_duplicates(subset=["id"], keep="last", inplace=True)
    else:
        combined_df = base_df

    # Merge IMDb Ratings & Vote Counts
    final_df = merge_imdb_ratings(combined_df)

    # ⚡ Filter out the 100,000 unwatched shows with 0 votes
    logger.info("Filtering for real/watched shows (vote_count >= 5 or imdb_votes >= 50)...")
    filtered_df = final_df[
        (final_df["vote_count"].fillna(0) >= 5) | (final_df["imdb_votes"].fillna(0) >= 50)
    ]

    logger.info(f"💾 Saving master TV dataset to {OUTPUT_CSV} ({len(filtered_df):,} shows)...")
    filtered_df.to_csv(OUTPUT_CSV, index=False)
    logger.info("🎉 Done!")


if __name__ == "__main__":
    main()
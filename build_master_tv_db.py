#!/usr/bin/env python3
"""
Master TMDb + IMDb TV Database Pipeline
- Ingests complete TV series & miniseries from official IMDb dumps
- Normalizes seasons & computes accurate 1-decimal place ratings/votes (Season 0 = Specials)
- Filters via Cinebetter-style TMDb cross-matching (drops unverified/podcast junk)
- Async parallelism (30 concurrent workers) with automatic rate-limit backoff
- Extracts backdrops, posters, plot, tagline, networks, studios, creators, cast with photos
- Ingests both TMDb & IMDb GraphQL 'More Like This' recommendations
- Resilient checkpointing (resumes if interrupted)
- Outputs directly into a normalized SQLite database (data/tv_master.db)
"""


import asyncio
import gzip
import json
import logging
import os
import sqlite3
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp
from dotenv import load_dotenv
import pandas as pd
from tqdm.asyncio import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("tv_master")

load_dotenv()
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
if not TMDB_API_KEY:
    logger.error("❌ TMDB_API_KEY environment variable required in .env file!")
    sys.exit(1)

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
SQLITE_DB = DATA_DIR / "tv_master.db"

IMDB_BASICS_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
IMDB_RATINGS_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"
IMDB_EPISODES_URL = "https://datasets.imdbws.com/title.episode.tsv.gz"

IMDB_GQL_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.3",
    "x-imdb-client-name": "imdb-web-next",
    "x-imdb-user-country": "US",
    "x-imdb-user-language": "en-US",
}

TMDB_HEADERS = {
    "Authorization": f"Bearer {TMDB_API_KEY}",
    "accept": "application/json",
}


# ==========================================
# 1. Database Setup & Initialization
# ==========================================
def init_database(db_path: Path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode = WAL;")
    cur.execute("PRAGMA synchronous = NORMAL;")
    cur.execute("PRAGMA foreign_keys = ON;")

    # 1. Master TV Shows Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tv_shows (
            imdb_id TEXT PRIMARY KEY,
            tmdb_id INTEGER,
            title TEXT NOT NULL,
            original_title TEXT,
            type TEXT,
            plot TEXT,
            tagline TEXT,
            poster_path TEXT,
            backdrop_path TEXT,
            start_year INTEGER,
            end_year INTEGER,
            runtime_minutes INTEGER,
            genres TEXT,
            status TEXT,
            networks TEXT,
            studios TEXT,
            imdb_rating REAL,
            imdb_votes INTEGER
        );
    """)

    # 2. Seasons Table (1-to-Many)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tv_seasons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            series_id TEXT NOT NULL,
            season_number INTEGER NOT NULL,
            season_name TEXT NOT NULL,
            episode_count INTEGER NOT NULL,
            rated_episodes INTEGER NOT NULL,
            avg_rating REAL,
            total_votes INTEGER,
            FOREIGN KEY (series_id) REFERENCES tv_shows(imdb_id) ON DELETE CASCADE
        );
    """)

    # 3. People Table (Cast & Showrunners)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tv_people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            series_id TEXT NOT NULL,
            person_type TEXT NOT NULL,
            name TEXT NOT NULL,
            character_name TEXT,
            image_url TEXT,
            display_order INTEGER,
            FOREIGN KEY (series_id) REFERENCES tv_shows(imdb_id) ON DELETE CASCADE
        );
    """)

    # 4. Recommendations Table (TMDb + IMDb)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tv_recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            series_id TEXT NOT NULL,
            recommended_imdb_id TEXT,
            recommended_title TEXT NOT NULL,
            source TEXT NOT NULL,
            rank_order INTEGER NOT NULL,
            FOREIGN KEY (series_id) REFERENCES tv_shows(imdb_id) ON DELETE CASCADE
        );
    """)

    conn.commit()
    conn.close()


def get_existing_imdb_ids(db_path: Path) -> Set[str]:
    if not db_path.exists():
        return set()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute("SELECT imdb_id FROM tv_shows;")
        rows = cur.fetchall()
        return {r[0] for r in rows}
    except Exception:
        return set()
    finally:
        conn.close()


# ==========================================
# 2. IMDb TSV Dumps Ingestion (Stage 1)
# ==========================================
def download_file(url: str, dest: Path):
    logger.info(f"📥 Downloading {dest.name}...")
    urllib.request.urlretrieve(url, dest)
    logger.info(f"✅ Downloaded {dest.name}")


def process_imdb_dumps() -> Tuple[pd.DataFrame, pd.DataFrame]:
    basics_gz = DATA_DIR / "title.basics.tsv.gz"
    ratings_gz = DATA_DIR / "title.ratings.tsv.gz"
    episodes_gz = DATA_DIR / "title.episode.tsv.gz"

    download_file(IMDB_BASICS_URL, basics_gz)
    download_file(IMDB_RATINGS_URL, ratings_gz)
    download_file(IMDB_EPISODES_URL, episodes_gz)

    logger.info("Extracting TV shows from title.basics.tsv.gz...")
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
    logger.info(f"Found {len(df_shows):,} total TV series/miniseries on IMDb.")

    logger.info("Loading ratings...")
    df_ratings = pd.read_csv(
        ratings_gz,
        sep="\t",
        na_values=r"\N",
        usecols=["tconst", "averageRating", "numVotes"],
        dtype={"tconst": str, "averageRating": float, "numVotes": float},
    )
    ratings_gz.unlink(missing_ok=True)

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
        },
        inplace=True,
    )

    df_shows["start_year"] = pd.to_numeric(df_shows["start_year"], errors="coerce")
    df_shows["end_year"] = pd.to_numeric(df_shows["end_year"], errors="coerce")
    df_shows["runtime_minutes"] = pd.to_numeric(
        df_shows["runtime_minutes"], errors="coerce"
    )
    df_shows["imdb_rating"] = df_shows["imdb_rating"].round(1)
    df_shows["imdb_votes"] = (
        pd.to_numeric(df_shows["imdb_votes"], errors="coerce").fillna(0).astype(int)
    )

    all_series_ids = set(df_shows["imdb_id"])

    logger.info("Aggregating seasons from title.episode.tsv.gz...")
    df_episodes = pd.read_csv(
        episodes_gz,
        sep="\t",
        na_values=r"\N",
        usecols=["tconst", "parentTconst", "seasonNumber"],
        dtype={"tconst": str, "parentTconst": str, "seasonNumber": str},
    )
    episodes_gz.unlink(missing_ok=True)

    df_episodes = df_episodes[df_episodes["parentTconst"].isin(all_series_ids)]
    df_episodes = df_episodes.merge(df_ratings, on="tconst", how="left")
    df_episodes["season_num"] = (
        pd.to_numeric(df_episodes["seasonNumber"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

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
    season_agg["season_name"] = season_agg["season_number"].apply(
        lambda s: "Specials" if s == 0 else f"Season {s}"
    )
    season_agg.sort_values(by=["series_id", "season_number"], inplace=True)

    return df_shows, season_agg


# ==========================================
# 3. Async TMDb & IMDb Enrichment
# ==========================================
async def fetch_tmdb_details(
    session: aiohttp.ClientSession, imdb_id: str, semaphore: asyncio.Semaphore
) -> Optional[Dict[str, Any]]:
    # 1. Find by External ID using v4 Bearer Auth
    find_url = f"https://api.themoviedb.org/3/find/{imdb_id}?external_source=imdb_id"
    async with semaphore:
        for _ in range(3):
            try:
                async with session.get(
                    find_url, headers=TMDB_HEADERS, timeout=12
                ) as res:
                    if res.status == 429:
                        retry_after = int(res.headers.get("Retry-After", 2))
                        await asyncio.sleep(retry_after)
                        continue
                    if res.status != 200:
                        return None
                    data = await res.json()
                    tv_results = data.get("tv_results") or []
                    if not tv_results:
                        return None
                    tmdb_id = tv_results[0]["id"]
                    break
            except Exception:
                await asyncio.sleep(1)
                continue
        else:
            return None

    # 2. Fetch Details using v4 Bearer Auth
    details_url = f"https://api.themoviedb.org/3/tv/{tmdb_id}?append_to_response=credits,recommendations"
    async with semaphore:
        for _ in range(3):
            try:
                async with session.get(
                    details_url, headers=TMDB_HEADERS, timeout=15
                ) as res:
                    if res.status == 429:
                        retry_after = int(res.headers.get("Retry-After", 2))
                        await asyncio.sleep(retry_after)
                        continue
                    if res.status != 200:
                        return None
                    return await res.json()
            except Exception:
                await asyncio.sleep(1)
                continue
        return None
    

async def fetch_imdb_recommendations(
    session: aiohttp.ClientSession, imdb_id: str, semaphore: asyncio.Semaphore
) -> List[Dict[str, Any]]:
    payload = {
        "query": "query MoreLikeThis($id: ID!) { title(id: $id) { moreLikeThisTitles(first: 10) { edges { node { id titleText { text } } } } } }",
        "variables": {"id": imdb_id},
    }
    async with semaphore:
        for _ in range(2):
            try:
                async with session.post(
                    "https://api.graphql.imdb.com/",
                    json=payload,
                    headers=IMDB_GQL_HEADERS,
                    timeout=10,
                ) as res:
                    if res.status != 200:
                        return []
                    data = await res.json()
                    edges = (
                        data.get("data", {})
                        .get("title", {})
                        .get("moreLikeThisTitles", {})
                        .get("edges")
                        or []
                    )
                    recs = []
                    for edge in edges:
                        node = edge.get("node") or {}
                        if node.get("id") and node.get("titleText", {}).get("text"):
                            recs.append(
                                {
                                    "id": node["id"],
                                    "title": node["titleText"]["text"],
                                }
                            )
                    return recs
            except Exception:
                await asyncio.sleep(0.5)
                continue
        return []


async def enrich_show(
    session: aiohttp.ClientSession,
    base_show: Dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> Optional[Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]]:
    imdb_id = base_show["imdb_id"]

    # Step A: TMDb verification and visual enrichment
    tmdb_data = await fetch_tmdb_details(session, imdb_id, semaphore)
    if not tmdb_data:
        # Filtered out (no TMDb match = low quality / non-standard)
        return None

    # Parse TMDb fields
    tmdb_id = tmdb_data.get("id")
    plot = tmdb_data.get("overview")
    tagline = tmdb_data.get("tagline")
    poster_path = tmdb_data.get("poster_path")
    backdrop_path = tmdb_data.get("backdrop_path")
    status = tmdb_data.get("status")

    networks = (
        ", ".join([n["name"] for n in tmdb_data.get("networks", []) if n.get("name")])
        or None
    )
    studios = (
        ", ".join(
            [
                c["name"]
                for c in tmdb_data.get("production_companies", [])
                if c.get("name")
            ]
        )
        or None
    )

    show_record = {
        "imdb_id": imdb_id,
        "tmdb_id": tmdb_id,
        "title": base_show["title"],
        "original_title": base_show["original_title"],
        "type": base_show["type"],
        "plot": plot,
        "tagline": tagline,
        "poster_path": poster_path,
        "backdrop_path": backdrop_path,
        "start_year": base_show["start_year"],
        "end_year": base_show["end_year"],
        "runtime_minutes": base_show["runtime_minutes"],
        "genres": base_show["genres"],
        "status": status,
        "networks": networks,
        "studios": studios,
        "imdb_rating": base_show["imdb_rating"],
        "imdb_votes": base_show["imdb_votes"],
    }

    # Step B: People (Creators + Top Cast)
    people_records = []
    # Creators
    for creator in tmdb_data.get("created_by", []):
        if creator.get("name"):
            people_records.append(
                {
                    "series_id": imdb_id,
                    "person_type": "creator",
                    "name": creator["name"],
                    "character_name": None,
                    "image_url": creator.get("profile_path"),
                    "display_order": 0,
                }
            )
    # Cast (Top 10)
    credits_cast = (tmdb_data.get("credits") or {}).get("cast", [])[:10]
    for idx, member in enumerate(credits_cast, start=1):
        if member.get("name"):
            people_records.append(
                {
                    "series_id": imdb_id,
                    "person_type": "cast",
                    "name": member["name"],
                    "character_name": member.get("character"),
                    "image_url": member.get("profile_path"),
                    "display_order": idx,
                }
            )

    # Step C: Recommendations (TMDb + IMDb)
    rec_records = []
    # TMDb Recommendations
    tmdb_recs = (tmdb_data.get("recommendations") or {}).get("results", [])[:10]
    for idx, r in enumerate(tmdb_recs, start=1):
        if r.get("name"):
            rec_records.append(
                {
                    "series_id": imdb_id,
                    "recommended_imdb_id": None,
                    "recommended_title": r["name"],
                    "source": "tmdb",
                    "rank_order": idx,
                }
            )

    # IMDb GraphQL Recommendations
    imdb_recs = await fetch_imdb_recommendations(session, imdb_id, semaphore)
    for idx, r in enumerate(imdb_recs, start=1):
        rec_records.append(
            {
                "series_id": imdb_id,
                "recommended_imdb_id": r["id"],
                "recommended_title": r["title"],
                "source": "imdb",
                "rank_order": idx,
            }
        )

    return show_record, people_records, rec_records


# ==========================================
# 4. Batch Pipeline Execution
# ==========================================
async def run_pipeline(shows_df: pd.DataFrame, seasons_df: pd.DataFrame, db_path: Path):
    init_database(db_path)
    existing_ids = get_existing_imdb_ids(db_path)
    if existing_ids:
        logger.info(
            f"⚡ Checkpoint detected: {len(existing_ids):,} shows already completed in SQLite."
        )

    # Filter out already completed shows
    pending_df = shows_df[~shows_df["imdb_id"].isin(existing_ids)]
    shows_list = pending_df.to_dict(orient="records")
    logger.info(
        f"🚀 Processing {len(shows_list):,} pending TV shows with 30 parallel workers..."
    )

    # Pre-write all seasons for pending shows into SQLite
    logger.info("Writing season structures to SQLite...")
    conn = sqlite3.connect(db_path)
    seasons_df.to_sql("tv_seasons", conn, if_exists="append", index=False)
    conn.close()

    semaphore = asyncio.Semaphore(30)
    batch_size = 300

    connector = aiohttp.TCPConnector(limit=100, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=connector) as session:
        for i in range(0, len(shows_list), batch_size):
            batch = shows_list[i : i + batch_size]
            tasks = [enrich_show(session, show, semaphore) for show in batch]
            results = await tqdm.gather(
                *tasks,
                desc=f"Batch {i // batch_size + 1}/{(len(shows_list) - 1) // batch_size + 1}",
            )

            shows_to_insert = []
            people_to_insert = []
            recs_to_insert = []

            for res in results:
                if res:
                    s_rec, p_recs, r_recs = res
                    shows_to_insert.append(s_rec)
                    people_to_insert.extend(p_recs)
                    recs_to_insert.extend(r_recs)

            # Write batch to SQLite
            if shows_to_insert:
                conn = sqlite3.connect(db_path)
                cur = conn.cursor()
                cur.execute("BEGIN TRANSACTION;")

                cur.executemany(
                    """
                    INSERT OR REPLACE INTO tv_shows (
                        imdb_id, tmdb_id, title, original_title, type, plot, tagline,
                        poster_path, backdrop_path, start_year, end_year, runtime_minutes,
                        genres, status, networks, studios, imdb_rating, imdb_votes
                    ) VALUES (
                        :imdb_id, :tmdb_id, :title, :original_title, :type, :plot, :tagline,
                        :poster_path, :backdrop_path, :start_year, :end_year, :runtime_minutes,
                        :genres, :status, :networks, :studios, :imdb_rating, :imdb_votes
                    );
                """,
                    shows_to_insert,
                )

                if people_to_insert:
                    cur.executemany(
                        """
                        INSERT INTO tv_people (series_id, person_type, name, character_name, image_url, display_order)
                        VALUES (:series_id, :person_type, :name, :character_name, :image_url, :display_order);
                    """,
                        people_to_insert,
                    )

                if recs_to_insert:
                    cur.executemany(
                        """
                        INSERT INTO tv_recommendations (series_id, recommended_imdb_id, recommended_title, source, rank_order)
                        VALUES (:series_id, :recommended_imdb_id, :recommended_title, :source, :rank_order);
                    """,
                        recs_to_insert,
                    )

                conn.commit()
                conn.close()

    # Create Indexes at the very end
    logger.info("Creating SQLite indexes...")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE INDEX IF NOT EXISTS idx_shows_title ON tv_shows(title);")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_shows_start_year ON tv_shows(start_year);"
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_shows_rating ON tv_shows(imdb_rating);")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_seasons_series ON tv_seasons(series_id);"
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_people_series ON tv_people(series_id);")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_recs_series ON tv_recommendations(series_id);"
    )
    conn.commit()
    conn.close()


def main():
    logger.info("🎬 Starting Master TV Database Build...")
    shows_df, seasons_df = process_imdb_dumps()
    asyncio.run(run_pipeline(shows_df, seasons_df, SQLITE_DB))
    logger.info(
        f"🎉 Master Database Build Complete! You can open '{SQLITE_DB}' in DB Browser for SQLite."
    )


if __name__ == "__main__":
    main()

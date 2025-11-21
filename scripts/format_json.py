import json
from pathlib import Path
import pandas as pd
import requests
import time

from config import get_auth_header

# ==== CONFIG ====

RAW_DIR = Path("./data/raw")
JSON_DIR = Path("./data/json")
JSON_DIR.mkdir(parents=True, exist_ok=True)

headers = get_auth_header()

IGDB_GENRES_URL = "https://api.igdb.com/v4/genres"
IGDB_PLATFORMS_URL = "https://api.igdb.com/v4/platforms"
IGDB_KEYWORDS_URL = "https://api.igdb.com/v4/keywords"
IGDB_INVOLVED_URL = "https://api.igdb.com/v4/involved_companies"
IGDB_COMPANIES_URL = "https://api.igdb.com/v4/companies"
IGDB_EXTERNAL_GAMES_URL = "https://api.igdb.com/v4/external_games"
STEAM_CATEGORY_ID = 1  # IGDB enum value for Steam

# ---- Fetch Steam appids via IGDB external_games ----

def fetch_steam_ids_for_games(game_ids):
    """
    Return a dict: { game_id (IGDB) -> steam_appid (string) }
    using IGDB's external_games endpoint, filtering category=Steam.
    """
    if not game_ids:
        return {}

    game_ids_list = sorted(list(game_ids))
    chunk_size = 500  # IGDB limit per request
    mapping = {}

    for i in range(0, len(game_ids_list), chunk_size):
        chunk = game_ids_list[i:i + chunk_size]
        ids_str = ",".join(str(gid) for gid in chunk)

        body = f"""
        fields game, uid, category;
        where game = ({ids_str}) & category = {STEAM_CATEGORY_ID};
        limit 500;
        """

        resp = requests.post(IGDB_EXTERNAL_GAMES_URL, headers=headers, data=body)
        resp.raise_for_status()
        items = resp.json()

        for item in items:
            game_id = item["game"]
            steam_uid = item["uid"]   # Steam appid as string/number
            # In case of multiple external entries per game, keep the first
            if game_id not in mapping:
                mapping[game_id] = str(steam_uid)

    return mapping

# ==== 1. LOAD RAW GAMES JSON ====

games_file = RAW_DIR / "games_raw.json"
with games_file.open("r", encoding="utf-8") as f:
    games = json.load(f)

print(f"Loaded {len(games)} games from {games_file}")

# collect all IGDB game_ids from raw games
game_ids_set = {g["id"] for g in games}

print("Fetching Steam appids from IGDB external_games...")
steam_ids_map = fetch_steam_ids_for_games(game_ids_set)
print(f"Found Steam appids for {len(steam_ids_map)} games")

# ==== 2. BUILD ROWS FOR TABLES ====

rows_games = []
rows_game_genres = []
rows_game_platforms = []
rows_game_keywords = []
rows_game_companies = []

genre_ids_set = set()
platform_ids_set = set()
keyword_ids_set = set()
involved_company_ids_set = set()
company_ids_set = set()

for g in games:
    game_id = g["id"]  # use IGDB id as our game_id

    rows_games.append({
        "game_id": game_id,
        "igdb_id": g["id"],
        "name": g.get("name"),
        "summary": g.get("summary"),
        "first_release_date": g.get("first_release_date"),
        "steam_appid": steam_ids_map.get(game_id),
    })

    # genres
    for genre_id in g.get("genres", []):
        rows_game_genres.append({
            "game_id": game_id,
            "genre_id": genre_id,
        })
        genre_ids_set.add(genre_id)

    # platforms
    for platform_id in g.get("platforms", []):
        rows_game_platforms.append({
            "game_id": game_id,
            "platform_id": platform_id,
        })
        platform_ids_set.add(platform_id)

    # keywords
    for keyword_id in g.get("keywords", []):
        rows_game_keywords.append({
            "game_id": game_id,
            "keyword_id": keyword_id,
        })
        keyword_ids_set.add(keyword_id)

    # involved_companies
    for inv_id in g.get("involved_companies", []):
        # we only know the involved_company ID for now
        rows_game_companies.append({
            "game_id": game_id,
            "involved_company_id": inv_id,
        })
        involved_company_ids_set.add(inv_id)

# Convert to DataFrames
df_games = pd.DataFrame(rows_games)
df_game_genres = pd.DataFrame(rows_game_genres)
df_game_platforms = pd.DataFrame(rows_game_platforms)


df_game_keywords = pd.DataFrame(rows_game_keywords)
df_game_companies = pd.DataFrame(rows_game_companies)

# ==== 3. OPTIONAL: RESOLVE GENRE / PLATFORM NAMES ====

def fetch_lookup_table(url, ids, id_field="id", name_field="name", chunk_size=500):
    """Fetch id->name mapping (genres, platforms, keywords, companies) from IGDB in chunks."""
    if not ids:
        return pd.DataFrame(columns=[id_field, name_field])

    ids_list = sorted(list(ids))
    frames = []

    for start in range(0, len(ids_list), chunk_size):
        chunk = ids_list[start:start + chunk_size]
        ids_str = ",".join(str(i) for i in chunk)

        body = f"""
        fields {id_field}, {name_field};
        where id = ({ids_str});
        limit {len(chunk)};
        """

        print(f"Fetching lookup from {url}: {start}–{start + len(chunk)} of {len(ids_list)}")
        resp = requests.post(url, headers=headers, data=body)

        if resp.status_code != 200:
            print("Error response:", resp.status_code, resp.text[:300])
            resp.raise_for_status()

        items = resp.json()
        if items:
            df_chunk = pd.DataFrame(items)[[id_field, name_field]]
            frames.append(df_chunk)

        # short delay to avoid hammering the API
        time.sleep(0.2)

    if frames:
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=[id_field])
    else:
        df = pd.DataFrame(columns=[id_field, name_field])

    return df

def fetch_involved_companies(ids, chunk_size=200):
    """
    Fetch involved_companies rows from IGDB in chunks to avoid 413 / payload too large.
    Returns a DataFrame with columns:
    id, company, game, developer, publisher, porting, supporting
    """
    if not ids:
        return pd.DataFrame(columns=["id", "company", "game", "developer", "publisher", "porting", "supporting"])

    ids_list = sorted(list(ids))
    frames = []

    for start in range(0, len(ids_list), chunk_size):
        chunk = ids_list[start:start + chunk_size]
        ids_str = ",".join(str(i) for i in chunk)

        body = f"""
        fields id, company, game, developer, publisher, porting, supporting;
        where id = ({ids_str});
        limit {len(chunk)};
        """

        print(f"Fetching involved_companies: {start}–{start + len(chunk)} of {len(ids_list)}")
        resp = requests.post(IGDB_INVOLVED_URL, headers=headers, data=body)

        if resp.status_code != 200:
            print("Error response from involved_companies:", resp.status_code, resp.text[:300])
            resp.raise_for_status()

        items = resp.json()
        if items:
            df_chunk = pd.DataFrame(items)
            frames.append(df_chunk)

        # be nice to IGDB
        time.sleep(0.2)

    if frames:
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["id"])
    else:
        df = pd.DataFrame(columns=["id", "company", "game", "developer", "publisher", "porting", "supporting"])

    return df


# genre lookup
df_genres = fetch_lookup_table(IGDB_GENRES_URL, genre_ids_set,
                               id_field="id", name_field="name")
df_genres = df_genres.rename(columns={"id": "genre_id"})

# platform lookup
df_platforms = fetch_lookup_table(IGDB_PLATFORMS_URL, platform_ids_set,
                                  id_field="id", name_field="name")
df_platforms = df_platforms.rename(columns={"id": "platform_id"})

# keyword lookup
df_keywords = fetch_lookup_table(IGDB_KEYWORDS_URL, keyword_ids_set,
                                 id_field="id", name_field="name")
df_keywords = df_keywords.rename(columns={"id": "keyword_id"})

df_involved = fetch_involved_companies(involved_company_ids_set)

# collect company IDs from here
if not df_involved.empty:
    company_ids_set = set(df_involved["company"].tolist())
else:
    company_ids_set = set()

df_companies = fetch_lookup_table(IGDB_COMPANIES_URL, company_ids_set,
                                  id_field="id", name_field="name")
df_companies = df_companies.rename(columns={"id": "company_id"})

# ==== 4. SAVE ALL JSONs ====

def save_df_as_json(df, path):
    records = df.to_dict(orient="records")
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

save_df_as_json(df_games, JSON_DIR / "games.json")
save_df_as_json(df_game_genres, JSON_DIR / "game_genres.json")
save_df_as_json(df_game_platforms, JSON_DIR / "game_platforms.json")
save_df_as_json(df_genres, JSON_DIR / "genres.json")
save_df_as_json(df_platforms, JSON_DIR / "platforms.json")

save_df_as_json(df_game_keywords, JSON_DIR / "game_keywords.json")
save_df_as_json(df_keywords, JSON_DIR / "keywords.json")
save_df_as_json(df_game_companies, JSON_DIR / "game_involved_companies.json")
save_df_as_json(df_involved, JSON_DIR / "involved_companies_raw.json")
save_df_as_json(df_companies, JSON_DIR / "companies.json")

print("Saved JSON files to", JSON_DIR)

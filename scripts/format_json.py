import json
from pathlib import Path
import pandas as pd
import requests

from config import get_auth_header

# ==== CONFIG ====

RAW_DIR = Path("./data/raw")
JSON_DIR = Path("./data/json")
JSON_DIR.mkdir(parents=True, exist_ok=True)

headers = get_auth_header()

IGDB_GENRES_URL = "https://api.igdb.com/v4/genres"
IGDB_PLATFORMS_URL = "https://api.igdb.com/v4/platforms"


# ==== 1. LOAD RAW GAMES JSON ====

games_file = RAW_DIR / "games_raw.json"
with games_file.open("r", encoding="utf-8") as f:
    games = json.load(f)

print(f"Loaded {len(games)} games from {games_file}")

# ==== 2. BUILD ROWS FOR TABLES ====

rows_games = []
rows_game_genres = []
rows_game_platforms = []

genre_ids_set = set()
platform_ids_set = set()

for g in games:
    game_id = g["id"]  # use IGDB id as our game_id

    rows_games.append({
        "game_id": game_id,
        "igdb_id": g["id"],
        "name": g.get("name"),
        "summary": g.get("summary"),
        "first_release_date": g.get("first_release_date"),
        "steam_appid": None,  # placeholder for future use
    })

    for genre_id in g.get("genres", []):
        rows_game_genres.append({
            "game_id": game_id,
            "genre_id": genre_id,
        })
        genre_ids_set.add(genre_id)

    for platform_id in g.get("platforms", []):
        rows_game_platforms.append({
            "game_id": game_id,
            "platform_id": platform_id,
        })
        platform_ids_set.add(platform_id)

# Convert to DataFrames
df_games = pd.DataFrame(rows_games)
df_game_genres = pd.DataFrame(rows_game_genres)
df_game_platforms = pd.DataFrame(rows_game_platforms)

# ==== 3. OPTIONAL: RESOLVE GENRE / PLATFORM NAMES ====

def fetch_lookup_table(url, ids, id_field="id", name_field="name"):
    """Fetch id->name mapping for genres or platforms from IGDB."""
    if not ids:
        return pd.DataFrame(columns=[id_field, name_field])

    ids_list = sorted(list(ids))
    # IGDB query: "where id = (1,2,3,...);"
    ids_str = ",".join(str(i) for i in ids_list)
    body = f"""
    fields {id_field}, {name_field};
    where id = ({ids_str});
    limit {len(ids_list)};
    """

    resp = requests.post(url, headers=headers, data=body)
    resp.raise_for_status()
    items = resp.json()
    return pd.DataFrame(items)[[id_field, name_field]]


# genre lookup
df_genres = fetch_lookup_table(IGDB_GENRES_URL, genre_ids_set,
                               id_field="id", name_field="name")
df_genres = df_genres.rename(columns={"id": "genre_id"})

# platform lookup
df_platforms = fetch_lookup_table(IGDB_PLATFORMS_URL, platform_ids_set,
                                  id_field="id", name_field="name")
df_platforms = df_platforms.rename(columns={"id": "platform_id"})

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

print("Saved JSON files to", JSON_DIR)

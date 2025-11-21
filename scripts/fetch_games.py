import requests
import json
from pathlib import Path

from config import get_auth_header

# ==== CONFIG ====

IGDB_GAMES_URL = "https://api.igdb.com/v4/games"

# Folder for raw data
RAW_DIR = Path("./data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ==== API CALL ====

headers = get_auth_header()

all_games = []
batch_size = 500
offset = 0

while True:
    print(f"Requesting games: offset={offset}, limit={batch_size}...")
    query = f"""
    fields id, name, summary, first_release_date, genres, platforms, keywords, involved_companies;
    where first_release_date != null;
    limit {batch_size};
    offset {offset};
    """

    resp = requests.post(IGDB_GAMES_URL, headers=headers, data=query)
    resp.raise_for_status()
    batch = resp.json()
    print(f"Got {len(batch)} games in this batch")

    if not batch:
        print("No more games, stopping.")
        break

    all_games.extend(batch)

    # move to next page
    offset += batch_size

    # optional: stop at some max, e.g. 5000 games for a school project
    # if offset >= 5000:
    #     break

print(f"Total games collected: {len(all_games)}")

out_file = RAW_DIR / "games_raw.json"
with out_file.open("w", encoding="utf-8") as f:
    json.dump(all_games, f, ensure_ascii=False, indent=2)

print(f"Saved all games to {out_file}")
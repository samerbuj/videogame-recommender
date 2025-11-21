import json
from pathlib import Path
import time
import requests

JSON_DIR = Path("./data/json")  # adjust if needed
GAMES_FILE = JSON_DIR / "games.json"
OUT_FILE = JSON_DIR / "steam_reviews.json"

STEAM_REVIEW_URL = (
    "https://store.steampowered.com/appreviews/{appid}"
    "?json=1&language=all&purchase_type=all&num_per_page=0"
)

# 1. Load games with steam_appid
with GAMES_FILE.open("r", encoding="utf-8") as f:
    games = json.load(f)

rows_reviews = []

for g in games:
    appid = g.get("steam_appid")
    game_id = g["game_id"]

    if not appid:
        continue  # skip games without steam mapping

    url = STEAM_REVIEW_URL.format(appid=appid)
    print(f"Fetching reviews for game_id={game_id}, appid={appid} ...")

    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()

        summary = data.get("query_summary", {})
        rows_reviews.append({
            "game_id": game_id,
            "steam_appid": appid,
            "total_positive": summary.get("total_positive", 0),
            "total_negative": summary.get("total_negative", 0),
            "total_reviews": summary.get("total_reviews", 0),
        })

        # small delay to be polite
        time.sleep(0.2)

    except Exception as e:
        print(f"Error for appid={appid}: {e}")

# 2. Save to JSON
with OUT_FILE.open("w", encoding="utf-8") as f:
    json.dump(rows_reviews, f, ensure_ascii=False, indent=2)

print(f"Saved Steam review stats to {OUT_FILE}")

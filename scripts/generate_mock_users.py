import json
import random
from pathlib import Path

# ==== CONFIG ====

JSON_DIR = Path("./data/json")
GAMES_FILE = JSON_DIR / "games.json"

USERS_OUT_FILE = JSON_DIR / "users.json"
USER_GAMES_OUT_FILE = JSON_DIR / "user_games.json"

N_USERS = 10000
MIN_GAMES_PER_USER = 1
MAX_GAMES_PER_USER = 50

# some random countries to add more stuff
COUNTRIES = ["US", "GB", "DE", "FR", "ES", "IT", "BR", "CA", "AU", "JP"]


# ==== 1. LOAD GAMES ====

with GAMES_FILE.open("r", encoding="utf-8") as f:
    games = json.load(f)

print(f"Loaded {len(games)} games from {GAMES_FILE}")

# Filter to games that have a steam_appid (optional, but better because of the reviews)
games_with_appid = [g for g in games if g.get("steam_appid")]
if games_with_appid:
    games = games_with_appid
    print(f"Using {len(games)} games that have a Steam appid")

if not games:
    raise RuntimeError("No games available to generate interactions!")

game_ids = [g["game_id"] for g in games]


# ==== 2. GENERATE MOCK USERS ====

def generate_user(user_id: int) -> dict:
    username = f"user_{user_id}"
    country = random.choice(COUNTRIES)
    age = random.randint(16, 45)  # arbitrary range

    return {
        "user_id": user_id,
        "username": username,
        "country": country,
        "age": age,
    }


users = [generate_user(i) for i in range(1, N_USERS + 1)]
print(f"Generated {len(users)} mock users")


# ==== 3. GENERATE USER-GAME INTERACTIONS ====

def generate_user_games(user_id: int) -> list[dict]:
    """
    Generate a random list of interactions for a single user.
    Each interaction has: user_id, game_id, rating, playtime_hours.
    """
    n_games = random.randint(MIN_GAMES_PER_USER, MAX_GAMES_PER_USER)
    # sample without replacement so each user doesn't repeat the same game
    sampled_game_ids = random.sample(game_ids, k=min(n_games, len(game_ids)))

    interactions = []
    for gid in sampled_game_ids:
        # rating: skew towards higher ratings a bit (gamers rarely rate 1/5)
        rating = round(min(max(random.gauss(3.8, 0.8), 1.0), 5.0), 1)

        # playtime: heavy-tailed – many low-play games, some very high
        # log-normal-ish: exp of a normal
        base = random.gauss(2.0, 1.0)
        playtime_hours = max(0.1, round((2.71828 ** base), 1))  # e^base, clamp at 0.1

        interactions.append({
            "user_id": user_id,
            "game_id": gid,
            "rating": rating,
            "playtime_hours": playtime_hours,
        })

    return interactions


user_games = []
for u in users:
    ug = generate_user_games(u["user_id"])
    user_games.extend(ug)

print(f"Generated {len(user_games)} user-game interactions")


# ==== 4. SAVE TO JSON ====

with USERS_OUT_FILE.open("w", encoding="utf-8") as f:
    json.dump(users, f, ensure_ascii=False, indent=2)

with USER_GAMES_OUT_FILE.open("w", encoding="utf-8") as f:
    json.dump(user_games, f, ensure_ascii=False, indent=2)

print(f"Saved users to {USERS_OUT_FILE}")
print(f"Saved user-game interactions to {USER_GAMES_OUT_FILE}")

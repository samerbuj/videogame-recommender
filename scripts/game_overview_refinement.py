import pandas as pd
from pathlib import Path
import json

from pathlib import Path
import pandas as pd

DATA_FOLDER = Path("./data/json") 

def generate_clean_game_overview():
    games = pd.read_json(DATA_FOLDER / "games.zip")
    genres = pd.read_json(DATA_FOLDER / "genres.json")
    platforms = pd.read_json(DATA_FOLDER / "platforms.json")
    companies = pd.read_json(DATA_FOLDER / "companies.json")
    keywords = pd.read_json(DATA_FOLDER / "keywords.json")
    game_genres = pd.read_json(DATA_FOLDER / "game_genres.json")
    game_platforms = pd.read_json(DATA_FOLDER / "game_platforms.json")
    game_involved = pd.read_json(DATA_FOLDER / "game_involved_companies.json")
    game_keywords = pd.read_json(DATA_FOLDER / "game_keywords.json")
    involved_raw = pd.read_json(DATA_FOLDER / "involved_companies_raw.json")
    reviews = pd.read_json(DATA_FOLDER / "steam_reviews.json")

    genre_map = dict(zip(genres.genre_id.astype(str), genres.name))
    platform_map = dict(zip(platforms.platform_id.astype(str), platforms.name))
    company_map = dict(zip(companies.company_id.astype(str), companies.name))
    keyword_map = dict(zip(keywords.keyword_id.astype(str), keywords.name))

    def join_names(df, id_col, map_dict):
        df = df.copy()
        df[id_col] = df[id_col].astype(str)
        df["name"] = df[id_col].map(map_dict)
        return (
            df.groupby("game_id")["name"]
            .apply(lambda x: ", ".join(sorted(set(x.dropna()))))
        )

    genre_text = join_names(game_genres, "genre_id", genre_map)
    platform_text = join_names(game_platforms, "platform_id", platform_map)
    keyword_text = join_names(game_keywords, "keyword_id", keyword_map)

    # Filter involved companies (require at least 1 True flag)
    involved_filtered = involved_raw[
        involved_raw[["developer", "porting", "publisher", "supporting"]].any(axis=1)
    ]
    involved_filtered["company"] = involved_filtered["company"].astype(str)
    involved_filtered["name"] = involved_filtered["company"].map(company_map)

    company_text = (
        involved_filtered
        .groupby("game")["name"]
        .apply(lambda x: ", ".join(sorted(set(x.dropna()))))
    )
    company_text.index.name = "game_id"

    # Merg into games
    df = games.copy()
    df["game_id"] = df["game_id"].astype(int)
    df = df.set_index("game_id")

    df["genres"] = genre_text
    df["platforms"] = platform_text
    df["companies"] = company_text
    df["keywords"] = keyword_text


    # Compute statistics of reviews and create rating scores
    reviews["game_id"] = reviews["game_id"].astype(int)
    reviews = reviews.set_index("game_id")

    mean_reviews = reviews["total_reviews"].mean()
    std_reviews = reviews["total_reviews"].std()
    median_reviews = reviews["total_reviews"].median()
    p25 = reviews["total_reviews"].quantile(0.25)
    p50 = reviews["total_reviews"].quantile(0.50)
    p75 = reviews["total_reviews"].quantile(0.75)
    p90 = reviews["total_reviews"].quantile(0.90)

    print("\n===== STATISTICS =====")
    print(f"Mean reviews: {mean_reviews:.2f}")
    print(f"Std dev: {std_reviews:.2f}")
    print(f"Median: {median_reviews:.0f}")
    print(f"25th percentile: {p25:.0f}")
    print(f"50th percentile: {p50:.0f}")
    print(f"75th percentile: {p75:.0f}")
    print(f"90th percentile: {p90:.0f}")
    print("=============================\n")

    min_reviews = 8
    valid_reviews = reviews[reviews["total_reviews"] >= min_reviews].copy()
    valid_reviews["steam_rating"] = valid_reviews["total_positive"] / valid_reviews["total_reviews"]
    df["steam_rating"] = df.index.map(valid_reviews["steam_rating"])
    df = df[df["steam_rating"].notna()]

    # Cleaning after merge
    # df = df.dropna(subset=["genres", "platforms", "companies"])
    df = df.dropna(subset=["genres"])

    # Date adjustments
    df["first_release_date"] = pd.to_datetime(df["first_release_date"], unit='s')
    df["first_release_date"] = df["first_release_date"].dt.strftime("%Y-%m-%d")

    # Final dataset
    df = df.reset_index()[
        [
            "game_id",
            "name",
            "summary",
            "first_release_date",
            "genres",
            "platforms",
            "companies",
            "keywords",
            "steam_rating"
        ]
    ]

    # Save
    out = DATA_FOLDER / "game_overview_final_vol2.json"
    df.to_json(out, orient="records", indent=2, force_ascii=False)
    print(f"Saved final merged clean dataset to: {out}")

    return df

if __name__ == "__main__":
    generate_clean_game_overview()
    df = pd.read_json(DATA_FOLDER / "game_overview_final_vol2.json")
    print(len(df))

    def inspect_nulls(df):
        # Columns to inspect
        cols = ["genres", "companies"]

        print("\n===== NULL COUNTS =====")
        total = len(df)
        print(f"Total entries: {total}\n")
        for col in cols:
            null_count = df[col].isna().sum()
            print(f"{col}: {null_count} nulls ({null_count/total*100:.2f}%)")

        print("\n===== COMBINED NULLS =====")

        # Null in genres AND companies
        both2 = df[df["genres"].isna() & df["companies"].isna()]
        print(f"Null in BOTH genres & companies: {len(both2)}")

        print("\n=========================\n")
    inspect_nulls(df)

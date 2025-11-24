
#import libraries
import json
from pathlib import Path
import pandas as pd

# GLOBAL VARIABLES
companies_id_mapping = {}
companies_null_ids = []
companies_ids = []
genres_id_mapping = {}
genres_null_ids = []
genres_ids = []
platforms_id_mapping = {}
platforms_null_ids = []
platforms_ids = []
keywords_id_mapping = {}
keywords_null_ids = []
keywords_ids = []

game_involved_companies_null_ids = []
game_genres_null_ids = []
game_platforms_null_ids = []
game_keywords_null_ids = []

# Base data folder
DATA_FOLDER = Path(__file__).resolve().parents[1] / "data" / "json"
OUTPUT_SUFFIX="_clean.json"

# Paths to individual JSON files
PATH_COMPANIES = DATA_FOLDER / "companies.json"
PATH_GENRES = DATA_FOLDER / "genres.json"
PATH_PLATFORMS = DATA_FOLDER / "platforms.json"
PATH_KEYWORDS = DATA_FOLDER / "keywords.json"

PATH_GAME_GENRES = DATA_FOLDER / "game_genres.json"
PATH_GAME_INVOLVED_COMPANIES = DATA_FOLDER / "game_involved_companies.json"
PATH_GAME_KEYWORDS = DATA_FOLDER / "game_keywords.json"
PATH_GAME_PLATFORMS = DATA_FOLDER / "game_platforms.json"

PATH_INVOLVED_COMPANIES_RAW = DATA_FOLDER / "involved_companies_raw.json"

PATH_STEAM_REVIEWS = DATA_FOLDER / "steam_reviews.json"

# Save Flag
SAVE_NEW_DATASET = False



# ==============================
# 1. COMPANIES
# ==============================

def clean_companies(input_path=PATH_COMPANIES, output_suffix=OUTPUT_SUFFIX):
    print("\n==============================")
    print("Cleaning companies.json")
    print("==============================")

    # Load JSON
    try:
        df = pd.read_json(input_path)
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        return


    print(f"Loaded {len(df)} entries")

    # Identify rows with missing values
    na_rows = df[df[["company_id", "name"]].isna().any(axis=1)]
    if not na_rows.empty:
        print("\nRows removed due to missing values:")
        print(na_rows)
        companies_null_ids.extend(na_rows["company_id"].dropna().astype(str).tolist())

    # Remove rows with missing company_id or name
    before_na = len(df)
    df = df.dropna(subset=["company_id", "name"])
    print(f"Removed {before_na - len(df)} rows with missing values")

    # Check for repeated IDs
    duplicate_id_rows = df[df["company_id"].duplicated(keep=False)]
    if not duplicate_id_rows.empty:
        print("\nRows with duplicated company_id:")
        print(duplicate_id_rows)
        df = df.drop_duplicates(subset=["company_id"])
    else:
        print("No duplicated company_id found")

    # Check for repeated names (ignoring case)
    df["name_lower"] = df["name"].str.lower()
    duplicate_name_rows = df[df["name_lower"].duplicated(keep=False)]
    if not duplicate_name_rows.empty:
        print("\nRows with duplicated names (case-insensitive):")
        print(duplicate_name_rows)
        grouped = duplicate_name_rows.groupby("name_lower")
        for _, group in grouped:
            if len(group) > 1:
                kept_id = str(group.iloc[0]["company_id"])
                for _, row in group.iloc[1:].iterrows():
                    companies_id_mapping[str(row["company_id"])] = kept_id
        df = df.drop_duplicates(subset=["name_lower"])
    else:
        print("No duplicated names found")

    df = df.drop(columns=["name_lower"])

    # Keep Ids
    rows = df[["company_id"]]
    if not rows.empty:
        companies_ids.extend(rows["company_id"].dropna().astype(str).tolist())

    print(f"Final cleaned entries: {len(df)}")

    print("\n--- Mappings of removed IDs to retained IDs (by name):")
    print(json.dumps(companies_id_mapping, indent=2))

    print("\n--- List of IDs removed due to missing name:")
    print(companies_null_ids)

    # Save cleaned file
    if SAVE_NEW_DATASET:
        output_path = input_path.parent / "clean" / (input_path.stem + output_suffix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(output_path, orient="records", indent=2)
        print(f"Saved cleaned file to: {output_path}\n")



# ==============================
# 2. GENRES
# ==============================

def clean_genres(input_path=PATH_GENRES, output_suffix=OUTPUT_SUFFIX):
    print("\n==============================")
    print("Cleaning genres.json")
    print("==============================")

    # Load JSON
    try:
        df = pd.read_json(input_path)
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        return

    print(f"Loaded {len(df)} entries")

    # Identify rows with missing values
    na_rows = df[df[["genre_id", "name"]].isna().any(axis=1)]
    if not na_rows.empty:
        print("\nRows removed due to missing values:")
        print(na_rows)
        genres_null_ids.extend(na_rows["genre_id"].dropna().astype(str).tolist())

    # Remove rows with missing genre_id or name
    before_na = len(df)
    df = df.dropna(subset=["genre_id", "name"])
    print(f"Removed {before_na - len(df)} rows with missing values")

    # Check for repeated IDs
    duplicate_id_rows = df[df["genre_id"].duplicated(keep=False)]
    if not duplicate_id_rows.empty:
        print("\nRows with duplicated genre_id:")
        print(duplicate_id_rows)
        df = df.drop_duplicates(subset=["genre_id"])
    else:
        print("No duplicated genre_id found")

    # Check for repeated names (ignoring case)
    df["name_lower"] = df["name"].str.lower()
    duplicate_name_rows = df[df["name_lower"].duplicated(keep=False)]
    if not duplicate_name_rows.empty:
        print("\nRows with duplicated names (case-insensitive):")
        print(duplicate_name_rows)
        grouped = duplicate_name_rows.groupby("name_lower")
        for _, group in grouped:
            if len(group) > 1:
                kept_id = str(group.iloc[0]["genre_id"])
                for _, row in group.iloc[1:].iterrows():
                    genres_id_mapping[str(row["genre_id"])] = kept_id
        df = df.drop_duplicates(subset=["name_lower"])
    else:
        print("No duplicated names found")

    df = df.drop(columns=["name_lower"])

    # Keep Ids
    rows = df[["genre_id"]]
    if not rows.empty:
        genres_ids.extend(rows["genre_id"].dropna().astype(str).tolist())

    print(f"Final cleaned entries: {len(df)}")

    print("\n--- Mappings of removed IDs to retained IDs (by name):")
    print(json.dumps(genres_id_mapping, indent=2))

    print("\n--- List of IDs removed due to missing name:")
    print(genres_null_ids)

    # Save cleaned file
    if SAVE_NEW_DATASET:
        output_path = input_path.parent / "clean" / (input_path.stem + output_suffix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(output_path, orient="records", indent=2)
        print(f"Saved cleaned file to: {output_path}\n")



# ==============================
# 3. PLATFORMS
# ==============================

def clean_platforms(input_path=PATH_PLATFORMS, output_suffix=OUTPUT_SUFFIX):
    print("\n==============================")
    print("Cleaning platforms.json")
    print("==============================")

    # Load JSON
    try:
        df = pd.read_json(input_path)
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        return

    print(f"Loaded {len(df)} entries")

    # Identify rows with missing values
    na_rows = df[df[["platform_id", "name"]].isna().any(axis=1)]
    if not na_rows.empty:
        print("\nRows removed due to missing values:")
        print(na_rows)
        platforms_null_ids.extend(na_rows["platform_id"].dropna().astype(str).tolist())

    # Remove rows with missing platform_id or name
    before_na = len(df)
    df = df.dropna(subset=["platform_id", "name"])
    print(f"Removed {before_na - len(df)} rows with missing values")

    # Check for repeated IDs
    duplicate_id_rows = df[df["platform_id"].duplicated(keep=False)]
    if not duplicate_id_rows.empty:
        print("\nRows with duplicated platform_id:")
        print(duplicate_id_rows)
        df = df.drop_duplicates(subset=["platform_id"])
    else:
        print("No duplicated platform_id found")

    # Check for repeated names (ignoring case)
    df["name_lower"] = df["name"].str.lower()
    duplicate_name_rows = df[df["name_lower"].duplicated(keep=False)]
    if not duplicate_name_rows.empty:
        print("\nRows with duplicated names (case-insensitive):")
        print(duplicate_name_rows)
        grouped = duplicate_name_rows.groupby("name_lower")
        for _, group in grouped:
            if len(group) > 1:
                kept_id = str(group.iloc[0]["platform_id"])
                for _, row in group.iloc[1:].iterrows():
                    platforms_id_mapping[str(row["platform_id"])] = kept_id
        df = df.drop_duplicates(subset=["name_lower"])
    else:
        print("No duplicated names found")

    df = df.drop(columns=["name_lower"])

    # Keep Ids
    rows = df[["platform_id"]]
    if not rows.empty:
        platforms_ids.extend(rows["platform_id"].dropna().astype(str).tolist())

    print(f"Final cleaned entries: {len(df)}")

    print("\n--- Mappings of removed IDs to retained IDs (by name):")
    print(json.dumps(platforms_id_mapping, indent=2))

    print("\n--- List of IDs removed due to missing name:")
    print(platforms_null_ids)

    # Save cleaned file
    if SAVE_NEW_DATASET:
        output_path = input_path.parent / "clean" / (input_path.stem + output_suffix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(output_path, orient="records", indent=2)
        print(f"Saved cleaned file to: {output_path}\n")



# ==============================
# 4. KEYWORDS
# ==============================

def clean_keywords(input_path=PATH_KEYWORDS, output_suffix=OUTPUT_SUFFIX):
    print("\n==============================")
    print("Cleaning keywords.json")
    print("==============================")

    # Load JSON
    try:
        df = pd.read_json(input_path)
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        return

    print(f"Loaded {len(df)} entries")

    # Identify rows with missing values
    na_rows = df[df[["keyword_id", "name"]].isna().any(axis=1)]
    if not na_rows.empty:
        print("\nRows removed due to missing values:")
        print(na_rows)
        keywords_null_ids.extend(na_rows["keyword_id"].dropna().astype(str).tolist())

    # Remove rows with missing keyword_id or name
    before_na = len(df)
    df = df.dropna(subset=["keyword_id", "name"])
    print(f"Removed {before_na - len(df)} rows with missing values")

    # Check for repeated IDs
    duplicate_id_rows = df[df["keyword_id"].duplicated(keep=False)]
    if not duplicate_id_rows.empty:
        print("\nRows with duplicated keyword_id:")
        print(duplicate_id_rows)
        df = df.drop_duplicates(subset=["keyword_id"])
    else:
        print("No duplicated keyword_id found")

    # Check for repeated names (ignoring case)
    df["name_lower"] = df["name"].str.lower()
    duplicate_name_rows = df[df["name_lower"].duplicated(keep=False)]
    if not duplicate_name_rows.empty:
        print("\nRows with duplicated names (case-insensitive):")
        print(duplicate_name_rows)
        grouped = duplicate_name_rows.groupby("name_lower")
        for _, group in grouped:
            if len(group) > 1:
                kept_id = str(group.iloc[0]["keyword_id"])
                for _, row in group.iloc[1:].iterrows():
                    keywords_id_mapping[str(row["keyword_id"])] = kept_id
        df = df.drop_duplicates(subset=["name_lower"])
    else:
        print("No duplicated names found")

    df = df.drop(columns=["name_lower"])

    # Keep Ids
    rows = df[["keyword_id"]]
    if not rows.empty:
        keywords_ids.extend(rows["keyword_id"].dropna().astype(str).tolist())

    print(f"Final cleaned entries: {len(df)}")

    print("\n--- Mappings of removed IDs to retained IDs (by name):")
    print(json.dumps(keywords_id_mapping, indent=2))

    print("\n--- List of IDs removed due to missing name:")
    print(keywords_null_ids)

    # Save cleaned file
    if SAVE_NEW_DATASET:
        output_path = input_path.parent / "clean" / (input_path.stem + output_suffix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(output_path, orient="records", indent=2)
        print(f"Saved cleaned file to: {output_path}\n")



# ==============================
# 5. GAME INVOLVED COMPANIES
# ==============================

def clean_game_involved_companies(input_path=PATH_GAME_INVOLVED_COMPANIES, output_suffix=OUTPUT_SUFFIX):
    print("\n==============================")
    print("Cleaning game_involved_companies.json")
    print("==============================")

    # Load JSON
    try:
        df = pd.read_json(input_path)
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        return

    print(f"Loaded {len(df)} entries")

    # Identify rows with missing values
    na_rows = df[df[["game_id", "involved_company_id"]].isna().any(axis=1)]
    if not na_rows.empty:
        print("\nRows removed due to missing values:")
        print(na_rows)
        game_involved_companies_null_ids.extend(na_rows["involved_company_id"].dropna().astype(str).tolist())

    # Remove rows with missing game_id or involved_company_id
    before_na = len(df)
    df = df.dropna(subset=["game_id", "involved_company_id"])
    print(f"Removed {before_na - len(df)} rows with missing values")

    # Convert to str for mapping and consistency
    df["involved_company_id"] = df["involved_company_id"].astype(str)
    df["game_id"] = df["game_id"].astype(int)

    # Apply mapping from companies.json cleaning
    if "companies_id_mapping" in globals() and companies_id_mapping:
        print("\nApplying companies_id_mapping to involved_company_id...")
        mapped_count = 0
        new_ids = []

        for cid in df["involved_company_id"]:
            new_id = companies_id_mapping.get(cid, cid)
            new_ids.append(new_id)
            if new_id != cid:
                mapped_count += 1

        df["involved_company_id"] = new_ids
        print(f"Mapped {mapped_count} involved_company_id values from old → new ID")

    # Drop duplicate rows
    duplicate_rows = df[df.duplicated(keep=False)]
    if not duplicate_rows.empty:
        print("\nDuplicated rows removed:")
        print(duplicate_rows)

    before_dedup = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before_dedup - len(df)} duplicated rows")

    print(f"Final cleaned entries: {len(df)}")

    print("\n--- List of involved_company_id removed due to missing fields:")
    print(game_involved_companies_null_ids)

    # Save cleaned file
    if SAVE_NEW_DATASET:
        output_path = input_path.parent / "clean" / (input_path.stem + output_suffix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(output_path, orient="records", indent=2)
        print(f"Saved cleaned file to: {output_path}\n")



# ==============================
# 6. GAME GENRES
# ==============================

def clean_game_genres(input_path=PATH_GAME_GENRES, output_suffix=OUTPUT_SUFFIX):
    print("\n==============================")
    print("Cleaning game_genres.json")
    print("==============================")

    # Load JSON
    try:
        df = pd.read_json(input_path)
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        return

    print(f"Loaded {len(df)} entries")

    # Identify rows with missing values
    na_rows = df[df[["game_id", "genre_id"]].isna().any(axis=1)]
    if not na_rows.empty:
        print("\nRows removed due to missing values:")
        print(na_rows)
        game_genres_null_ids.extend(na_rows["genre_id"].dropna().astype(str).tolist())

    # Remove rows with missing game_id or genre_id
    before_na = len(df)
    df = df.dropna(subset=["game_id", "genre_id"])
    print(f"Removed {before_na - len(df)} rows with missing values")

    # Convert to str for mapping and consistency
    df["genre_id"] = df["genre_id"].astype(str)
    df["game_id"] = df["game_id"].astype(int)

    # Apply mapping from genres.json cleaning
    if "genres_id_mapping" in globals() and genres_id_mapping:
        print("\nApplying genres_id_mapping to genre_id...")
        mapped_count = 0
        new_ids = []

        for cid in df["genre_id"]:
            new_id = genres_id_mapping.get(cid, cid)
            new_ids.append(new_id)
            if new_id != cid:
                mapped_count += 1

        df["genre_id"] = new_ids
        print(f"Mapped {mapped_count} genre_id values from old → new ID")

    # Drop duplicate rows
    duplicate_rows = df[df.duplicated(keep=False)]
    if not duplicate_rows.empty:
        print("\nDuplicated rows removed:")
        print(duplicate_rows)

    before_dedup = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before_dedup - len(df)} duplicated rows")

    print(f"Final cleaned entries: {len(df)}")

    print("\n--- List of genre_id removed due to missing fields:")
    print(game_genres_null_ids)

    # Save cleaned file
    if SAVE_NEW_DATASET:
        output_path = input_path.parent / "clean" / (input_path.stem + output_suffix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(output_path, orient="records", indent=2)
        print(f"Saved cleaned file to: {output_path}\n")



# ==============================
# 7. GAME PLATFORMS
# ==============================

def clean_game_platforms(input_path=PATH_GAME_PLATFORMS, output_suffix=OUTPUT_SUFFIX):
    print("\n==============================")
    print("Cleaning game_platforms.json")
    print("==============================")

    # Load JSON
    try:
        df = pd.read_json(input_path)
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        return

    print(f"Loaded {len(df)} entries")

    # Identify rows with missing values
    na_rows = df[df[["game_id", "platform_id"]].isna().any(axis=1)]
    if not na_rows.empty:
        print("\nRows removed due to missing values:")
        print(na_rows)
        game_platforms_null_ids.extend(na_rows["platform_id"].dropna().astype(str).tolist())

    # Remove rows with missing game_id or platform_id
    before_na = len(df)
    df = df.dropna(subset=["game_id", "platform_id"])
    print(f"Removed {before_na - len(df)} rows with missing values")

    # Convert to str for mapping and consistency
    df["platform_id"] = df["platform_id"].astype(str)
    df["game_id"] = df["game_id"].astype(int)

    # Apply mapping from platforms.json cleaning
    if "platforms_id_mapping" in globals() and platforms_id_mapping:
        print("\nApplying platforms_id_mapping to platform_id...")
        mapped_count = 0
        new_ids = []

        for cid in df["platform_id"]:
            new_id = platforms_id_mapping.get(cid, cid)
            new_ids.append(new_id)
            if new_id != cid:
                mapped_count += 1

        df["platform_id"] = new_ids
        print(f"Mapped {mapped_count} platform_id values from old → new ID")

    # Drop duplicate rows
    duplicate_rows = df[df.duplicated(keep=False)]
    if not duplicate_rows.empty:
        print("\nDuplicated rows removed:")
        print(duplicate_rows)

    before_dedup = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before_dedup - len(df)} duplicated rows")

    print(f"Final cleaned entries: {len(df)}")

    print("\n--- List of platform_id removed due to missing fields:")
    print(game_platforms_null_ids)

    # Save cleaned file
    if SAVE_NEW_DATASET:
        output_path = input_path.parent / "clean" / (input_path.stem + output_suffix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(output_path, orient="records", indent=2)
        print(f"Saved cleaned file to: {output_path}\n")



# ==============================
# 8. GAME KEYWORDS
# ==============================

def clean_game_keywords(input_path=PATH_GAME_KEYWORDS, output_suffix=OUTPUT_SUFFIX):
    print("\n==============================")
    print("Cleaning game_keywords.json")
    print("==============================")

    # Load JSON
    try:
        df = pd.read_json(input_path)
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        return

    print(f"Loaded {len(df)} entries")

    # Identify rows with missing values
    na_rows = df[df[["game_id", "keyword_id"]].isna().any(axis=1)]
    if not na_rows.empty:
        print("\nRows removed due to missing values:")
        print(na_rows)
        game_keywords_null_ids.extend(na_rows["keyword_id"].dropna().astype(str).tolist())

    # Remove rows with missing game_id or keyword_id
    before_na = len(df)
    df = df.dropna(subset=["game_id", "keyword_id"])
    print(f"Removed {before_na - len(df)} rows with missing values")

    # Convert to str for mapping and consistency
    df["keyword_id"] = df["keyword_id"].astype(str)
    df["game_id"] = df["game_id"].astype(int)

    # Apply mapping from keywords.json cleaning
    if "keywords_id_mapping" in globals() and keywords_id_mapping:
        print("\nApplying keywords_id_mapping to keyword_id...")
        mapped_count = 0
        new_ids = []

        for cid in df["keyword_id"]:
            new_id = keywords_id_mapping.get(cid, cid)
            new_ids.append(new_id)
            if new_id != cid:
                mapped_count += 1

        df["keyword_id"] = new_ids
        print(f"Mapped {mapped_count} keyword_id values from old → new ID")

    # Drop duplicate rows
    duplicate_rows = df[df.duplicated(keep=False)]
    if not duplicate_rows.empty:
        print("\nDuplicated rows removed:")
        print(duplicate_rows)

    before_dedup = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before_dedup - len(df)} duplicated rows")

    print(f"Final cleaned entries: {len(df)}")

    print("\n--- List of keyword_id removed due to missing fields:")
    print(game_keywords_null_ids)

    # Save cleaned file
    if SAVE_NEW_DATASET:
        output_path = input_path.parent / "clean" / (input_path.stem + output_suffix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(output_path, orient="records", indent=2)
        print(f"Saved cleaned file to: {output_path}\n")



# ==============================
# 9. INVOLVED COMPANIES RAW
# ==============================

def clean_involved_companies_raw(input_path=PATH_INVOLVED_COMPANIES_RAW, output_suffix=OUTPUT_SUFFIX):
    print("\n==============================")
    print("Cleaning involved_companies_raw.json")
    print("==============================")

    # Load JSON
    try:
        df = pd.read_json(input_path)
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        return

    print(f"Loaded {len(df)} entries")

    # Remove duplicated rows
    before_dupes = len(df)
    dupes = df[df.duplicated(keep=False)]
    if not dupes.empty:
        print("\nDuplicated rows removed (examples):")
        print(dupes.head(5))
    df = df.drop_duplicates()
    print(f"Removed {before_dupes - len(df)} duplicated rows")

    # Drop rows where company does not exist in companies_ids
    before_company = len(df)
    missing_company_mask = ~df["company"].astype(str).isin(companies_ids)
    if missing_company_mask.any():
        print(f"\nRows removed due to non-existent company (examples):")
        print(df[missing_company_mask].head(5))
    df = df[~missing_company_mask]
    print(f"Removed {before_company - len(df)} rows due to non-existent company values")

    print(f"Final cleaned entries: {len(df)}")

    # Save cleaned file
    if SAVE_NEW_DATASET:
        output_path = input_path.parent / "clean" / (input_path.stem + output_suffix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(output_path, orient="records", indent=2)
        print(f"Saved cleaned file to: {output_path}\n")



# ==============================
# 10. STEAM REVIEWS
# ==============================

def clean_steam_reviews(input_path=PATH_STEAM_REVIEWS, output_suffix=OUTPUT_SUFFIX):
    print("\n==============================")
    print("Cleaning steam_reviews.json")
    print("==============================")

    # Load JSON
    try:
        df = pd.read_json(input_path)
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        return

    print(f"Loaded {len(df)} entries")

    # Remove rows where all three columns are zero
    zero_mask = (df["total_positive"] == 0) & (df["total_negative"] == 0) & (df["total_reviews"] == 0)
    zero_rows = df[zero_mask]
    if not zero_rows.empty:
        print(f"\nRows removed where all review counts are zero (examples):")
        print(zero_rows.head(5))
    df = df[~zero_mask]
    print(f"Removed {len(zero_rows)} rows where all review counts are zero")

    # Remove rows where total_reviews != total_positive + total_negative
    wrong_sum_mask = df["total_reviews"] != (df["total_positive"] + df["total_negative"])
    wrong_sum_rows = df[wrong_sum_mask]
    if not wrong_sum_rows.empty:
        print(f"\nRows removed where total_reviews != total_positive + total_negative (examples):")
        print(wrong_sum_rows.head(5))
    df = df[~wrong_sum_mask]
    print(f"Removed {len(wrong_sum_rows)} rows where total_reviews did not match sum")

    print(f"Final cleaned entries: {len(df)}")

    # Save cleaned file
    if SAVE_NEW_DATASET:
        output_path = input_path.parent / "clean" / (input_path.stem + output_suffix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(output_path, orient="records", indent=2)
        print(f"Saved cleaned file to: {output_path}\n")




# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    clean_companies()
    clean_genres()
    clean_platforms()
    clean_keywords()
    clean_game_involved_companies()
    clean_game_genres()
    clean_game_platforms()
    clean_game_keywords()
    clean_involved_companies_raw()
    clean_steam_reviews()
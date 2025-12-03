import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from pathlib import Path


# =========================
# 1. Load and prepare data
# =========================

def load_user_game_ratings(path: str) -> pd.DataFrame:
    """
    Load user_games.json into a pandas DataFrame.
    Expected keys: user_id, game_id, rating (playtime_hours is ignored here).
    """
    df = pd.read_json(path)
    df = df[["user_id", "game_id", "rating"]].dropna()
    return df


def encode_ids(df: pd.DataFrame):
    """
    Map original user_id and game_id to 0..num-1 indices.

    Returns:
      - df_encoded with columns: user_idx, item_idx, rating
      - user_id_to_idx, idx_to_user_id
      - game_id_to_idx, idx_to_game_id
    """
    # Sort for reproducibility
    user_ids = sorted(df["user_id"].unique())
    game_ids = sorted(df["game_id"].unique())

    user_id_to_idx: Dict[int, int] = {uid: i for i, uid in enumerate(user_ids)}
    idx_to_user_id: Dict[int, int] = {i: uid for uid, i in user_id_to_idx.items()}

    game_id_to_idx: Dict[int, int] = {gid: i for i, gid in enumerate(game_ids)}
    idx_to_game_id: Dict[int, int] = {i: gid for gid, i in game_id_to_idx.items()}

    df_encoded = pd.DataFrame({
        "user_idx": df["user_id"].map(user_id_to_idx),
        "item_idx": df["game_id"].map(game_id_to_idx),
        "rating": df["rating"].astype(float)
    })

    return df_encoded, user_id_to_idx, idx_to_user_id, game_id_to_idx, idx_to_game_id


def train_test_split_interactions(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Random train/test split on interactions.
    df must have columns: user_idx, item_idx, rating.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)

    test_size = int(len(df) * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    train = df.iloc[train_idx].to_numpy()
    test = df.iloc[test_idx].to_numpy()

    return train, test


def load_game_metadata(path: str) -> pd.DataFrame:
    """
    Load game_overview.json.
    Keeps basic metadata and converts release date to datetime.
    """
    games = pd.read_json(path)

    # Convert ms-since-epoch to datetime if present
    if "first_release_date" in games.columns:
        games["first_release_date"] = pd.to_datetime(
            games["first_release_date"], unit="ms", errors="coerce"
        )

    keep_cols = [
        "game_id",
        "name",
        "summary",
        "first_release_date",
        "genres",
        "platforms",
        "companies",
        "keywords",
        "steam_rating",
    ]
    games = games[[c for c in keep_cols if c in games.columns]]

    return games


# =========================
# 2. Matrix Factorization
# =========================

class MatrixFactorization:
    def __init__(
        self,
        num_users: int,
        num_items: int,
        n_factors: int = 20,
        lr: float = 0.01,
        reg: float = 0.1,
        n_epochs: int = 10,
        seed: int = 42
    ):
        """
        Basic MF model: R ~ P @ Q^T with user/item biases.
        """
        self.num_users = num_users
        self.num_items = num_items
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.rng = np.random.default_rng(seed)

        # Latent factors
        self.P = 0.1 * self.rng.standard_normal((num_users, n_factors))
        self.Q = 0.1 * self.rng.standard_normal((num_items, n_factors))

        # Bias terms
        self.user_bias = np.zeros(num_users)
        self.item_bias = np.zeros(num_items)
        self.global_mean = 0.0

    def fit(self, train_data: np.ndarray, verbose: bool = True):
        """
        Train using SGD.
        train_data: array of shape (N, 3) with columns [user_idx, item_idx, rating].
        """
        self.global_mean = train_data[:, 2].mean()

        for epoch in range(self.n_epochs):
            self.rng.shuffle(train_data)
            epoch_loss = 0.0

            for u_idx, i_idx, r in train_data:
                u = int(u_idx)
                i = int(i_idx)
                r = float(r)

                pred = self.predict_single(u, i)
                err = r - pred

                pu = self.P[u, :].copy()
                qi = self.Q[i, :].copy()

                # Bias updates
                self.user_bias[u] += self.lr * (err - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * (err - self.reg * self.item_bias[i])

                # Latent factor updates
                self.P[u, :] += self.lr * (err * qi - self.reg * pu)
                self.Q[i, :] += self.lr * (err * pu - self.reg * qi)

                epoch_loss += err ** 2

            rmse_val = np.sqrt(epoch_loss / len(train_data))
            if verbose:
                print(f"Epoch {epoch + 1}/{self.n_epochs}, RMSE (train): {rmse_val:.4f}")

    def predict_single(self, user_idx: int, item_idx: int) -> float:
        """
        Predict rating for a single (user, item) pair.
        """
        pred = (
            self.global_mean
            + self.user_bias[user_idx]
            + self.item_bias[item_idx]
            + np.dot(self.P[user_idx, :], self.Q[item_idx, :])
        )
        return float(pred)

    def predict_batch(self, data: np.ndarray) -> np.ndarray:
        """
        Predict ratings for an array of [user_idx, item_idx, rating] triplets.
        (rating column is ignored)
        """
        preds = np.zeros(len(data), dtype=float)
        for idx, (u_idx, i_idx, _) in enumerate(data):
            preds[idx] = self.predict_single(int(u_idx), int(i_idx))
        return preds

    def recommend_for_user(
        self,
        user_idx: int,
        known_item_indices: List[int],
        top_n: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Recommend top_n new items for user_idx.
        known_item_indices: items the user already interacted with (to exclude).
        Returns list of (item_idx, predicted_rating) sorted by rating desc.
        """
        all_items = np.arange(self.num_items)
        mask = np.ones(self.num_items, dtype=bool)
        mask[known_item_indices] = False
        candidate_items = all_items[mask]

        scores: List[Tuple[int, float]] = []
        for i in candidate_items:
            scores.append((i, self.predict_single(user_idx, i)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]


# =========================
# 3. Evaluation helper
# =========================

def rmse(model: MatrixFactorization, test_data: np.ndarray) -> float:
    preds = model.predict_batch(test_data)
    true = test_data[:, 2].astype(float)
    mse = np.mean((true - preds) ** 2)
    return float(np.sqrt(mse))


# =========================
# 4. Main
# =========================

if __name__ == "__main__":
    # --- Build paths relative to THIS FOLDER (where the script lives) ---
    BASE_DIR = Path(__file__).resolve().parent

    # Expect user_games.json and (optionally) game_overview.json in the same folder
    user_games_path = BASE_DIR / "data" / "json" / "user_games.json"
    game_overview_path = BASE_DIR / "data" / "json" / "clean" / "game_overview_final_vol2.json"

    # 1) Load interactions
    if not user_games_path.is_file():
        raise FileNotFoundError(
            f"user_games.json not found at {user_games_path}. "
            f"Put user_games.json in the same folder as this script."
        )

    df = load_user_game_ratings(str(user_games_path))
    print("Loaded interactions:", len(df))

    # 2) Encode ids
    (
        df_encoded,
        user_id_to_idx,
        idx_to_user_id,
        game_id_to_idx,
        idx_to_game_id,
    ) = encode_ids(df)

    num_users = len(user_id_to_idx)
    num_items = len(game_id_to_idx)
    print(f"Num users: {num_users}, Num items: {num_items}")

    # 3) Train/test split
    train_data, test_data = train_test_split_interactions(df_encoded, test_ratio=0.2)
    print(f"Train interactions: {len(train_data)}, Test interactions: {len(test_data)}")

    # 4) Load game metadata (optional)
    if game_overview_path.is_file():
        game_meta_df = load_game_metadata(str(game_overview_path))
        game_meta_dict = game_meta_df.set_index("game_id").to_dict(orient="index")
        print("Loaded game_overview.json for metadata.")
    else:
        game_meta_dict = {}
        print("Warning: game_overview.json not found in this folder.")
        print("Recommendations will only show game_id and predicted_rating.")

    # 5) Train the MF model
    model = MatrixFactorization(
        num_users=num_users,
        num_items=num_items,
        n_factors=40,
        lr=0.01,
        reg=0.05,
        n_epochs=10,
    )
    model.fit(train_data, verbose=True)

    # 6) Evaluate
    test_rmse_val = rmse(model, test_data)
    print(f"Test RMSE: {test_rmse_val:.4f}")

    # 7) Precompute which items each user has already interacted with
    user_items = (
        df_encoded.groupby("user_idx")["item_idx"].apply(list).to_dict()
    )

    # 8) Generate recommendations for ALL users
    all_rows = []

    for user_idx, user_id in idx_to_user_id.items():
        known_items = user_items.get(user_idx, [])
        recs = model.recommend_for_user(
            user_idx=user_idx,
            known_item_indices=known_items,
            top_n=10,
        )

        for item_idx, score in recs:
            game_id = idx_to_game_id[item_idx]
            meta = game_meta_dict.get(game_id, {})
            name = meta.get("name", "<unknown>")
            genres = meta.get("genres", "")
            platforms = meta.get("platforms", "")
            steam_rating = meta.get("steam_rating", None)
            release_date = meta.get("first_release_date", None)

            all_rows.append({
                "user_id": user_id,
                "game_id": game_id,
                "name": name,
                "predicted_rating": score,
                "steam_rating": steam_rating,
                "genres": genres,
                "platforms": platforms,
                "first_release_date": release_date,
            })

    # 9) Save all users' recommendations to CSV
    all_df = pd.DataFrame(all_rows)
    output_path = BASE_DIR / "collaborative_filtering_output.csv"
    all_df.to_csv(output_path, index=False)
    print(f"Saved recommendations for all users to: {output_path}")

    input_path = Path("collaborative_filtering_output.csv")
    output_path = Path("collaborative_filtering_output_clean.csv")

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load CSV
    df = pd.read_csv(input_path)

    if "name" not in df.columns:
        raise KeyError(
            "Column 'name' not found in the CSV. "
            "Check the column names in your file."
        )

    # Keep only rows where name is not '<unknown>' and not empty/NaN
    mask = (df["name"].notna()) & (df["name"].astype(str).str.strip() != "<unknown>")
    df_clean = df[mask].copy()

    print(f"Original rows: {len(df)}")
    print(f"Rows after removing '<unknown>': {len(df_clean)}")

    # Save cleaned CSV
    df_clean.to_csv(output_path, index=False)
    print(f"Cleaned file saved to: {output_path}")
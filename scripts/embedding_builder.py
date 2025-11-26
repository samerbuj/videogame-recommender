# %%
# Get the embeddings

import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import re
import time
import tracemalloc

tracemalloc.start()
t0 = time.perf_counter()

# Load dataset
df = pd.read_json("data/game_overview/game_overview.json")
df = df.reset_index(drop=True)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace(";", " ")
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


#Build the text to embed
def make_text(row):
    parts = []
    
    for col in ['name', 'summary', 'genres', 'platforms', 'companies', 'keywords']:
        if col in row:
            val = row[col]
            if isinstance(val, str) and val.strip():
                parts.append(val)

    return " ".join(parts)

df['text'] = df.apply(make_text, axis=1)

print(df[['game_id', 'name', 'text']].head())
print(df.shape)


# %%
#Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

#Encode to embedding
texts = df["text"].tolist()
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

print(embeddings.shape)

np.save("embeddings.npy", embeddings)
df.to_pickle("games_df.pkl")  # preserves dataframe structure, strings, lists, etc.

print("Saved embeddings.npy and games_df.pkl")
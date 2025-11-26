# %%
# Get the embeddings

import os

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

df = pd.read_pickle("games_df.pkl")
embeddings = np.load("embeddings.npy")
print("Embeddings shape:", embeddings.shape)
print("Number of games:", len(df))


# %%
# generate random hyperplanes

def generate_hyperplanes(num_tables, num_planes, dim, seed= 0):
    rng = np.random.RandomState(seed)
    hyperplanes = []
    for _ in range(num_tables):
        planes = rng.randn(num_planes, dim)
        planes /= np.linalg.norm(planes, axis = 1, keepdims = True) +1e-9
        hyperplanes.append(planes)
    
    return hyperplanes

# check it
dim = embeddings.shape[1]
num_tables = 10
num_planes = 16

hyperplanes = generate_hyperplanes(num_tables, num_planes, dim, seed=33)
print(hyperplanes)


# %%
# Hash each embedding into buckets 
def hash_v(v, planes):
    projections = planes @ v
    bits = projections > 0
    key = ''.join('1' if b else '0' for b in bits)
    return key

# %%
# Get LSH index, which is tables with buckets

def build_lsh_idx(embeddings, hyperplanes):
    n, dim = embeddings.shape
    num_tables = len(hyperplanes)

    tables = [defaultdict(set) for _ in range(num_tables)]


    for idx, v in enumerate(embeddings):
        for t, planes in enumerate(hyperplanes):
            key = hash_v(v, planes)
            tables[t][key].add(idx)
    
    return tables

# check it
tables = build_lsh_idx(embeddings, hyperplanes)


# %%
# query LSH; get candidate neighbours for a game
def lsh_query(idx, embeddings, hyperplanes, tables):
    v = embeddings[idx]
    candidates = set()

    for t, planes in enumerate(hyperplanes):
        key = hash_v(v, planes)
        bucket = tables[t].get(key, set())
        candidates |= bucket

    candidates.discard(idx)
    return candidates

#test
cand = lsh_query(5, embeddings, hyperplanes, tables)
print("Candidates for game 0:", cand)
for candidate in cand:
    print(df.loc[candidate, 'name'])

# %%
# Rank candidates by cosine similarity
def recommend_lsh_embedding(idx, embeddings, hyperplanes, tables, df, top=5):
    v = embeddings[idx]
    cand = lsh_query(idx, embeddings, hyperplanes, tables)

    if not cand:
        print(f"No candidates for {df.loc[idx, 'name']}")
        return []

    cand_indices = np.array(list(cand), dtype=int)

    sim = cosine_similarity(
        v.reshape(1, -1),
        embeddings[cand_indices]
    )  # shape: (1, n_candidates)

    sim_flat = sim[0]  
    cos_norm = (sim_flat + 1.0) / 2.0


    ratings = df.loc[cand_indices, 'steam_rating'].astype(float).to_numpy()

    alpha = 0.8
    final_score = alpha * cos_norm + (1.0 - alpha) * ratings
    
    order = np.argsort(final_score)[::-1]
    ordered_indices = cand_indices[order]
    ordered_cos_norm = cos_norm[order]    
    ordered_rating = ratings[order]
    ordered_score = final_score[order]

    results = list(zip(
        ordered_indices[:top], 
        ordered_cos_norm[:top],
        ordered_rating[:top],
        ordered_score[:top]))

    print(f"Query: {df.loc[idx, 'name']}")
    print("Recommendations (cosine_norm, rating combined):")

    for j, cos_s, r, score in results:
        print(
            f" --> {df.loc[j, 'name']} "
            f"(cos={cos_s:.3f}, rating={r:.3f}, score={score:.3f})"
        )

    return results


#test
recommend_lsh_embedding(5, embeddings, hyperplanes, tables, df, top=3)


elapsed = time.perf_counter() - t0
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"Total runtime of entire script: {elapsed:.4f} seconds")
print(f"Peak memory usage: {peak / (1024**2):.2f} MB")



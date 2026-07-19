"""
train_phase1_embeddings.py
==========================
PHASE 1 (upgraded): 'Honest' answerability checker using SENTENCE EMBEDDINGS.

Why embeddings: SQuAD 2.0's unanswerable questions reuse the passage's words
on purpose, so word-overlap fails (~50%). Embeddings compare MEANING, not
words, which is what's needed to tell them apart.

Pipeline:
  question + context
        |
        v
  embed question, embed each context sentence   (sentence-transformers)
        |
        v
  features: max similarity, mean of top-k sims, gap, etc.
        |  (+ a couple of cheap word-overlap features)
        v
  Logistic Regression  ->  answerable (1) / unanswerable (0)

First run downloads a small model (~90 MB) from the internet, then works offline.

Outputs:
  - accuracy / precision / recall / F1 on held-out test data
  - saves studymate_model_emb.joblib  (model + config) for the app
  - caches embeddings so re-runs are fast
"""

import re
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
import joblib

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-mpnet-base-v2"   # small, fast, good quality, beginner standard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_sentences(text):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(text)) if s.strip()]
    return sents if sents else [str(text)]


def cosine(a, b):
    # a: (d,)  b: (n, d)  -> (n,)   assumes rows are L2-normalised
    return b @ a


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

def build_features(df, embedder, cache_path=None):
    """
    For each (question, context):
      embed the question and every context sentence, then summarise how
      semantically close the best sentences are. These similarity stats are
      the features the classifier learns from.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"  loading cached features from {cache_path}")
        return np.load(cache_path)

    feats = []
    questions = df["question"].tolist()
    contexts = df["context"].tolist()

    # embed all questions at once (fast)
    print("  embedding questions...")
    q_emb = embedder.encode(questions, normalize_embeddings=True,
                            show_progress_bar=False, batch_size=64)

    print("  embedding contexts sentence-by-sentence...")
    for i, ctx in enumerate(contexts):
        sents = split_sentences(ctx)
        s_emb = embedder.encode(sents, normalize_embeddings=True,
                                show_progress_bar=False, batch_size=64)
        sims = cosine(q_emb[i], s_emb)          # similarity to each sentence
        sims_sorted = np.sort(sims)[::-1]

        max_sim = float(sims_sorted[0])
        top2_mean = float(np.mean(sims_sorted[:2]))
        mean_sim = float(np.mean(sims))
        gap = float(sims_sorted[0] - (sims_sorted[1] if len(sims_sorted) > 1 else 0))
        n_strong = float(np.sum(sims > 0.5))    # how many clearly-relevant sentences

        feats.append([max_sim, top2_mean, mean_sim, gap, n_strong])

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(contexts)} contexts done")

    arr = np.array(feats, dtype=float)
    if cache_path:
        np.save(cache_path, arr)
    return arr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(csv_path="studymate_data.csv"):
    df = pd.read_csv(csv_path).reset_index(drop=True)
    print(f"Loaded {len(df)} rows")

    print(f"Loading embedding model '{MODEL_NAME}' (first run downloads it)...")
    embedder = SentenceTransformer(MODEL_NAME)

    print("Building features (this is the slow part, then it's cached)...")
    X = build_features(df, embedder, cache_path="features_emb.npy")
    y = df["answerable"].values

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(Xtr)}   Test: {len(Xte)}")

    model = LogisticRegression(max_iter=1000)
    model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    print("\n=== RESULTS ON UNSEEN TEST DATA ===")
    print(f"Accuracy : {accuracy_score(yte, pred):.3f}")
    print(f"Precision: {precision_score(yte, pred):.3f}")
    print(f"Recall   : {recall_score(yte, pred):.3f}")
    print(f"F1 score : {f1_score(yte, pred):.3f}")
    cm = confusion_matrix(yte, pred)
    print("\nConfusion matrix:")
    print("                 predicted_unans  predicted_ans")
    print(f"  actual_unans        {cm[0][0]:5d}          {cm[0][1]:5d}")
    print(f"  actual_ans          {cm[1][0]:5d}          {cm[1][1]:5d}")

    joblib.dump({"model": model, "model_name": MODEL_NAME}, "studymate_model_emb.joblib")
    print("\nSaved trained model to studymate_model_emb.joblib")


if __name__ == "__main__":
    main()

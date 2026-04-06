"""
services/recommender.py
========================
TF-IDF Content-Based Recommendation Engine

This module encapsulates the existing TF-IDF logic into a clean service class.
It computes cosine similarity between movie feature vectors built from metadata
(genres, keywords, cast, overview, etc.) that were pre-computed and stored in
tfidf_matrix.pkl.

How it works:
1. Each movie is represented as a TF-IDF vector of its combined text features.
2. Given a query movie, we compute dot-product similarity against all others.
3. Results are sorted by descending similarity score.
4. The top-N most similar movies (excluding the query itself) are returned.
"""

import os
import pickle
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd


class ContentRecommender:
    """
    Content-based recommender using pre-computed TF-IDF vectors.
    
    Attributes:
        df:           DataFrame with movie metadata (must have 'title' column)
        tfidf_matrix: Sparse matrix of TF-IDF vectors (n_movies x n_features)
        title_to_idx: Normalized title -> row index mapping
        idx_to_title: Row index -> original title mapping
    """

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.df: Optional[pd.DataFrame] = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.title_to_idx: Dict[str, int] = {}
        self.idx_to_title: Dict[int, str] = {}
        self._loaded = False

    def load(self):
        """Load all pickle files and build lookup maps."""
        df_path = os.path.join(self.base_dir, "df.pkl")
        indices_path = os.path.join(self.base_dir, "indices.pkl")
        tfidf_matrix_path = os.path.join(self.base_dir, "tfidf_matrix.pkl")
        tfidf_path = os.path.join(self.base_dir, "tfidf.pkl")

        with open(df_path, "rb") as f:
            self.df = pickle.load(f)

        with open(indices_path, "rb") as f:
            indices_obj = pickle.load(f)

        with open(tfidf_matrix_path, "rb") as f:
            self.tfidf_matrix = pickle.load(f)

        with open(tfidf_path, "rb") as f:
            self.tfidf_vectorizer = pickle.load(f)

        # Build normalized title <-> index maps
        self.title_to_idx = self._build_title_map(indices_obj)
        self.idx_to_title = {v: k for k, v in self.title_to_idx.items()}

        # Validate
        if self.df is None or "title" not in self.df.columns:
            raise RuntimeError("df.pkl must contain a DataFrame with 'title' column")

        self._loaded = True
        print(f"[ContentRecommender] Loaded {len(self.title_to_idx)} movies")

    def _build_title_map(self, indices: Any) -> Dict[str, int]:
        """
        Convert indices.pkl (dict or pandas Series) into a normalized
        lowercase title -> integer index mapping.
        """
        title_map: Dict[str, int] = {}
        try:
            for k, v in indices.items():
                title_map[str(k).strip().lower()] = int(v)
        except Exception:
            raise RuntimeError("indices.pkl must be dict or Series with .items()")
        return title_map

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_index(self, title: str) -> Optional[int]:
        """Look up the matrix row index for a given title (case-insensitive)."""
        return self.title_to_idx.get(title.strip().lower())

    def get_title(self, idx: int) -> str:
        """Get the original title for a matrix row index."""
        if self.df is not None:
            try:
                return str(self.df.iloc[idx]["title"])
            except (IndexError, KeyError):
                pass
        return self.idx_to_title.get(idx, f"Unknown-{idx}")

    def get_movie_genres(self, title: str) -> List[str]:
        """
        Extract genre names for a given title from the DataFrame.
        Handles both string and list genre columns gracefully.
        """
        if self.df is None:
            return []
        idx = self.get_index(title)
        if idx is None:
            return []
        try:
            row = self.df.iloc[idx]
            genres_raw = row.get("genres", "")
            if isinstance(genres_raw, list):
                # List of dicts: [{"id": 28, "name": "Action"}, ...]
                return [g["name"] if isinstance(g, dict) else str(g) for g in genres_raw]
            if isinstance(genres_raw, str) and genres_raw:
                # Could be space-separated or comma-separated
                return [g.strip() for g in genres_raw.replace(",", " ").split() if g.strip()]
        except Exception:
            pass
        return []

    def recommend(
        self, title: str, top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get top-N content-based recommendations for a movie title.

        Returns:
            List of dicts: [{"title": str, "score": float, "index": int}, ...]
        
        Raises:
            ValueError if title not found in dataset.
        """
        if not self._loaded:
            raise RuntimeError("ContentRecommender not loaded. Call .load() first.")

        idx = self.get_index(title)
        if idx is None:
            raise ValueError(f"Title not found in local dataset: '{title}'")

        # Compute cosine similarity: query vector dot all vectors
        # tfidf_matrix is sparse, so this is efficient
        query_vec = self.tfidf_matrix[idx]
        scores = (self.tfidf_matrix @ query_vec.T).toarray().ravel()

        # Sort descending, skip self
        order = np.argsort(-scores)

        results: List[Dict[str, Any]] = []
        for i in order:
            i = int(i)
            if i == idx:
                continue
            t = self.get_title(i)
            results.append({
                "title": t,
                "score": float(scores[i]),
                "index": i,
            })
            if len(results) >= top_n:
                break

        return results

    def recommend_multi(
        self, titles: List[str], top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Aggregate content-based recommendations across multiple seed titles.
        Useful for watchlist-based recommendations.
        
        Strategy: average the TF-IDF vectors of all seed movies, then find
        the most similar movies to this "user profile" vector.
        """
        if not self._loaded:
            raise RuntimeError("ContentRecommender not loaded.")

        # Collect valid indices
        seed_indices = []
        for t in titles:
            idx = self.get_index(t)
            if idx is not None:
                seed_indices.append(idx)

        if not seed_indices:
            return []

        # Build aggregated profile vector (mean of seed vectors)
        profile = self.tfidf_matrix[seed_indices].mean(axis=0)
        # profile is a numpy matrix (1 x n_features), convert for dot product
        profile = np.asarray(profile).ravel()

        # Score all movies against the profile
        scores = (self.tfidf_matrix @ profile).ravel()
        if hasattr(scores, 'A1'):
            # scipy sparse result
            scores = scores.A1
        scores = np.asarray(scores).ravel()

        order = np.argsort(-scores)

        seed_set = set(seed_indices)
        results: List[Dict[str, Any]] = []
        for i in order:
            i = int(i)
            if i in seed_set:
                continue
            results.append({
                "title": self.get_title(i),
                "score": float(scores[i]),
                "index": i,
            })
            if len(results) >= top_n:
                break

        return results

    def get_all_titles(self) -> List[str]:
        """Return all titles in the dataset."""
        if self.df is not None:
            return self.df["title"].tolist()
        return list(self.idx_to_title.values())
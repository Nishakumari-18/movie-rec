"""
services/collaborative.py
===========================
Collaborative Filtering Engine using Truncated SVD

This module implements user-based collaborative filtering:

1. BUILD a user-item interaction matrix from tracked behaviors:
   - Watchlist additions  → implicit rating 5.0
   - Movie clicks/views   → implicit rating 3.0
   - Search interactions   → implicit rating 1.0

2. FACTORIZE using Truncated SVD (Singular Value Decomposition):
   - Decomposes the sparse user-item matrix into latent factors
   - U (users x k), Σ (k x k), Vt (k x items)
   - Predicted ratings = U @ Σ @ Vt

3. PREDICT: For a given user, predict scores for unseen movies
   and return the highest-scoring ones as recommendations.

Why SVD?
- Handles the sparsity problem well (most users interact with few movies)
- Discovers latent features (e.g., "dark thrillers", "feel-good comedies")
- Scales better than raw user-user cosine similarity
- Standard technique expected in academic ML projects
"""

import os
import json
import time
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


# ─── Interaction weights (implicit feedback) ───
WEIGHT_WATCHLIST = 5.0
WEIGHT_CLICK = 3.0
WEIGHT_SEARCH = 1.0


class CollaborativeEngine:
    """
    SVD-based collaborative filtering engine.

    Maintains a user-item interaction store and rebuilds the SVD model
    on demand (or periodically). Designed for a moderate number of users
    typical in a demo/academic project (< 10K users).
    """

    def __init__(self, data_dir: str, n_factors: int = 50):
        """
        Args:
            data_dir:   Directory to store/load user_interactions.json
            n_factors:  Number of latent factors for SVD decomposition
        """
        self.data_dir = data_dir
        self.interactions_path = os.path.join(data_dir, "user_interactions.json")
        self.n_factors = n_factors

        # Raw interaction data: {user_email: {movie_title: score}}
        self.interactions: Dict[str, Dict[str, float]] = {}

        # Mappings built during model fitting
        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        self.movie_to_idx: Dict[str, int] = {}
        self.idx_to_movie: Dict[int, str] = {}

        # SVD components
        self.U: Optional[np.ndarray] = None  # users x factors
        self.sigma: Optional[np.ndarray] = None  # factors
        self.Vt: Optional[np.ndarray] = None  # factors x items
        self.predicted_ratings: Optional[np.ndarray] = None

        self._model_built = False
        self._last_build_time = 0.0

    # ═══════════════════════════════════════════════
    # PERSISTENCE: Load / Save interactions
    # ═══════════════════════════════════════════════

    def load_interactions(self):
        """Load interaction data from JSON file."""
        if os.path.exists(self.interactions_path):
            try:
                with open(self.interactions_path, "r") as f:
                    self.interactions = json.load(f)
                print(f"[CollabEngine] Loaded interactions for {len(self.interactions)} users")
            except (json.JSONDecodeError, IOError):
                self.interactions = {}
        else:
            self.interactions = {}

    def save_interactions(self):
        """Persist interaction data to JSON file."""
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self.interactions_path, "w") as f:
            json.dump(self.interactions, f, indent=2)

    # ═══════════════════════════════════════════════
    # RECORD interactions (called from API routes)
    # ═══════════════════════════════════════════════

    def record_interaction(
        self, user_email: str, movie_title: str, interaction_type: str
    ):
        """
        Record a user-movie interaction.
        
        Args:
            interaction_type: "watchlist", "click", or "search"
        
        Uses max() so stronger interactions don't get downgraded.
        """
        weight_map = {
            "watchlist": WEIGHT_WATCHLIST,
            "click": WEIGHT_CLICK,
            "search": WEIGHT_SEARCH,
        }
        weight = weight_map.get(interaction_type, WEIGHT_CLICK)
        norm_title = movie_title.strip().lower()

        if user_email not in self.interactions:
            self.interactions[user_email] = {}

        # Keep the maximum weight (watchlist > click > search)
        current = self.interactions[user_email].get(norm_title, 0.0)
        self.interactions[user_email][norm_title] = max(current, weight)

        self.save_interactions()
        # Invalidate model so it rebuilds on next recommendation
        self._model_built = False

    def record_watchlist_bulk(self, user_email: str, titles: List[str]):
        """Record watchlist items in bulk (e.g., on login sync)."""
        if user_email not in self.interactions:
            self.interactions[user_email] = {}
        for t in titles:
            norm = t.strip().lower()
            current = self.interactions[user_email].get(norm, 0.0)
            self.interactions[user_email][norm] = max(current, WEIGHT_WATCHLIST)
        self.save_interactions()
        self._model_built = False

    # ═══════════════════════════════════════════════
    # BUILD SVD MODEL
    # ═══════════════════════════════════════════════

    def build_model(self, force: bool = False):
        """
        Build/rebuild the SVD model from current interactions.
        
        Steps:
        1. Create user and movie index mappings
        2. Build sparse user-item matrix
        3. Run Truncated SVD
        4. Compute full predicted ratings matrix
        
        Skips rebuild if model is fresh (< 60s old) unless force=True.
        """
        if self._model_built and not force:
            if time.time() - self._last_build_time < 60:
                return  # model is fresh enough

        n_users = len(self.interactions)
        if n_users == 0:
            self._model_built = False
            return

        # Step 1: Build index mappings
        all_movies: set = set()
        for user_movies in self.interactions.values():
            all_movies.update(user_movies.keys())

        self.user_to_idx = {u: i for i, u in enumerate(sorted(self.interactions.keys()))}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
        self.movie_to_idx = {m: i for i, m in enumerate(sorted(all_movies))}
        self.idx_to_movie = {i: m for m, i in self.movie_to_idx.items()}

        n_movies = len(self.movie_to_idx)

        if n_users < 2 or n_movies < 2:
            # SVD needs at least 2x2 matrix; fall back to non-SVD
            self._model_built = False
            return

        # Step 2: Build sparse matrix
        rows, cols, vals = [], [], []
        for user, movies in self.interactions.items():
            uid = self.user_to_idx[user]
            for movie, score in movies.items():
                if movie in self.movie_to_idx:
                    mid = self.movie_to_idx[movie]
                    rows.append(uid)
                    cols.append(mid)
                    vals.append(score)

        matrix = csr_matrix(
            (vals, (rows, cols)),
            shape=(n_users, n_movies),
            dtype=np.float32,
        )

        # Step 3: Truncated SVD
        # k must be < min(n_users, n_movies)
        k = min(self.n_factors, n_users - 1, n_movies - 1, 50)
        if k < 1:
            self._model_built = False
            return

        try:
            U, sigma, Vt = svds(matrix.astype(float), k=k)
            self.U = U
            self.sigma = sigma
            self.Vt = Vt

            # Step 4: Predicted ratings = U @ diag(sigma) @ Vt
            sigma_diag = np.diag(sigma)
            self.predicted_ratings = U @ sigma_diag @ Vt

            self._model_built = True
            self._last_build_time = time.time()
            print(f"[CollabEngine] SVD model built: {n_users} users × {n_movies} movies, k={k}")

        except Exception as e:
            print(f"[CollabEngine] SVD failed: {e}")
            self._model_built = False

    # ═══════════════════════════════════════════════
    # RECOMMEND
    # ═══════════════════════════════════════════════

    def recommend(
        self, user_email: str, top_n: int = 10, exclude_seen: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get collaborative filtering recommendations for a user.

        Returns:
            List of {"title": str, "score": float}
            Scores are predicted ratings from the SVD model.
        """
        self.build_model()

        if not self._model_built or self.predicted_ratings is None:
            return self._fallback_recommend(user_email, top_n)

        if user_email not in self.user_to_idx:
            return []

        uid = self.user_to_idx[user_email]
        pred_row = self.predicted_ratings[uid]

        # Get user's already-seen movies
        seen_indices = set()
        if exclude_seen and user_email in self.interactions:
            for movie in self.interactions[user_email]:
                if movie in self.movie_to_idx:
                    seen_indices.add(self.movie_to_idx[movie])

        # Rank unseen movies by predicted score
        candidates = []
        for mid in range(len(pred_row)):
            if mid in seen_indices:
                continue
            candidates.append((mid, float(pred_row[mid])))

        candidates.sort(key=lambda x: -x[1])

        results = []
        for mid, score in candidates[:top_n]:
            title = self.idx_to_movie.get(mid, f"movie-{mid}")
            results.append({"title": title, "score": score})

        return results

    def _fallback_recommend(
        self, user_email: str, top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fallback when SVD can't be built (too few users/items).
        Uses simple co-occurrence: find movies liked by users who
        share the most overlap with the target user.
        """
        if user_email not in self.interactions:
            return []

        user_movies = set(self.interactions[user_email].keys())
        if not user_movies:
            return []

        # Find similar users by Jaccard overlap
        user_scores: List[Tuple[str, float]] = []
        for other, other_movies_dict in self.interactions.items():
            if other == user_email:
                continue
            other_movies = set(other_movies_dict.keys())
            overlap = len(user_movies & other_movies)
            if overlap > 0:
                jaccard = overlap / len(user_movies | other_movies)
                user_scores.append((other, jaccard))

        user_scores.sort(key=lambda x: -x[1])

        # Collect movies from similar users that target hasn't seen
        movie_agg: Dict[str, float] = defaultdict(float)
        for other_user, sim in user_scores[:20]:
            for movie, weight in self.interactions[other_user].items():
                if movie not in user_movies:
                    movie_agg[movie] += sim * weight

        ranked = sorted(movie_agg.items(), key=lambda x: -x[1])
        return [{"title": t, "score": s} for t, s in ranked[:top_n]]

    def get_user_interactions(self, user_email: str) -> Dict[str, float]:
        """Get raw interaction scores for a user."""
        return dict(self.interactions.get(user_email, {}))

    def get_similar_users(self, user_email: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Find users with similar taste (for explainability / metrics).
        Uses latent factor similarity from SVD when available.
        """
        if not self._model_built or self.U is None:
            return []
        if user_email not in self.user_to_idx:
            return []

        uid = self.user_to_idx[user_email]
        user_vec = self.U[uid]

        # Cosine similarity in latent space
        norms = np.linalg.norm(self.U, axis=1)
        norms[norms == 0] = 1e-10
        user_norm = np.linalg.norm(user_vec)
        if user_norm == 0:
            return []

        similarities = (self.U @ user_vec) / (norms * user_norm)
        order = np.argsort(-similarities)

        results = []
        for i in order:
            i = int(i)
            if i == uid:
                continue
            results.append({
                "user": self.idx_to_user.get(i, f"user-{i}"),
                "similarity": float(similarities[i]),
            })
            if len(results) >= top_n:
                break
        return results

    @property
    def stats(self) -> Dict[str, Any]:
        """Return model statistics for the metrics page."""
        n_users = len(self.interactions)
        n_interactions = sum(
            len(movies) for movies in self.interactions.values()
        )
        all_movies = set()
        for movies in self.interactions.values():
            all_movies.update(movies.keys())

        return {
            "total_users": n_users,
            "total_interactions": n_interactions,
            "unique_movies_interacted": len(all_movies),
            "model_built": self._model_built,
            "n_factors": self.n_factors,
            "matrix_shape": (
                f"{n_users} x {len(all_movies)}" if n_users > 0 else "N/A"
            ),
            "sparsity": (
                f"{(1 - n_interactions / max(n_users * len(all_movies), 1)) * 100:.1f}%"
                if n_users > 0 and all_movies else "N/A"
            ),
        }
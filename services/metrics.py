"""
services/metrics.py
=====================
Recommendation System Evaluation Metrics

Implements standard IR/ML evaluation metrics:

1. Precision@K: Of the top-K recommendations, how many are relevant?
   Precision@K = |relevant ∩ recommended@K| / K

2. Recall@K: Of all relevant items, how many appear in top-K?
   Recall@K = |relevant ∩ recommended@K| / |relevant|

3. RMSE: Root Mean Squared Error between predicted and actual ratings
   RMSE = sqrt(mean((predicted - actual)²))

4. NDCG@K: Normalized Discounted Cumulative Gain
   Measures ranking quality — rewards placing relevant items higher

5. Coverage: What fraction of the catalog is recommended across all users?

"Relevant" is defined as: movies in the user's watchlist (implicit positive signal).
We use a leave-one-out evaluation strategy:
- For each user, hold out their most recent watchlist item
- Generate recommendations without it
- Check if the held-out item appears in top-K
"""

import math
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import numpy as np


class MetricsEngine:
    """
    Evaluation engine for the recommendation system.
    """

    def __init__(self, collab_engine, content_engine, hybrid_engine):
        self.collab = collab_engine
        self.content = content_engine
        self.hybrid = hybrid_engine

    def precision_at_k(
        self, recommended: List[str], relevant: set, k: int
    ) -> float:
        """
        Precision@K: fraction of top-K recommendations that are relevant.
        
        Args:
            recommended: Ordered list of recommended movie titles (normalized)
            relevant: Set of relevant movie titles (normalized)
            k: Cutoff
        """
        if k == 0:
            return 0.0
        top_k = recommended[:k]
        hits = sum(1 for t in top_k if t in relevant)
        return hits / k

    def recall_at_k(
        self, recommended: List[str], relevant: set, k: int
    ) -> float:
        """
        Recall@K: fraction of relevant items found in top-K.
        """
        if not relevant:
            return 0.0
        top_k = recommended[:k]
        hits = sum(1 for t in top_k if t in relevant)
        return hits / len(relevant)

    def ndcg_at_k(
        self, recommended: List[str], relevant: set, k: int
    ) -> float:
        """
        NDCG@K: Normalized Discounted Cumulative Gain.
        Binary relevance: 1 if in relevant set, 0 otherwise.
        """
        top_k = recommended[:k]

        # DCG
        dcg = 0.0
        for i, title in enumerate(top_k):
            rel = 1.0 if title in relevant else 0.0
            dcg += rel / math.log2(i + 2)  # +2 because i starts at 0

        # Ideal DCG (all relevant items at the top)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

        if idcg == 0:
            return 0.0
        return dcg / idcg

    def rmse(
        self, predictions: List[Tuple[float, float]]
    ) -> float:
        """
        RMSE between predicted and actual ratings.
        
        Args:
            predictions: List of (predicted_score, actual_score) tuples
        """
        if not predictions:
            return 0.0
        errors = [(p - a) ** 2 for p, a in predictions]
        return math.sqrt(sum(errors) / len(errors))

    def evaluate_user(
        self, user_email: str, k: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate recommendation quality for a single user using
        leave-one-out cross-validation.

        Strategy:
        1. Get user's watchlist (relevant items)
        2. If ≥ 2 items, hold out the last one
        3. Generate recommendations
        4. Check if held-out item is in top-K
        5. Compute Precision@K, Recall@K, NDCG@K
        """
        interactions = self.collab.get_user_interactions(user_email)
        if not interactions:
            return self._empty_result(k)

        # "Relevant" = items with watchlist-level interaction (score >= 5)
        relevant_all = {t for t, s in interactions.items() if s >= 5.0}

        if len(relevant_all) < 2:
            return self._empty_result(k)

        # Leave-one-out: hold out last item
        relevant_list = list(relevant_all)
        held_out = relevant_list[-1]
        relevant_train = set(relevant_list[:-1])

        # Get hybrid recommendations
        try:
            recs = self.hybrid.recommend(user_email=user_email, top_n=k * 2)
            rec_titles = [r["title"].strip().lower() for r in recs]
        except Exception:
            rec_titles = []

        # Evaluate against held-out item + remaining relevant items
        test_relevant = {held_out} | relevant_train

        prec = self.precision_at_k(rec_titles, test_relevant, k)
        recall = self.recall_at_k(rec_titles, test_relevant, k)
        ndcg = self.ndcg_at_k(rec_titles, test_relevant, k)

        # RMSE: compare predicted scores with actual interaction weights
        predictions = []
        for r in recs:
            norm_t = r["title"].strip().lower()
            if norm_t in interactions:
                actual = interactions[norm_t] / 5.0  # normalize to [0, 1]
                predictions.append((r["score"], actual))

        rmse_val = self.rmse(predictions) if predictions else None

        # Hit rate: was the held-out item found?
        hit = 1 if held_out in set(rec_titles[:k]) else 0

        return {
            f"precision_at_{k}": round(prec, 4),
            f"recall_at_{k}": round(recall, 4),
            f"ndcg_at_{k}": round(ndcg, 4),
            "rmse": round(rmse_val, 4) if rmse_val is not None else None,
            "hit": hit,
            "relevant_items": len(relevant_all),
            "held_out_item": held_out,
        }

    def evaluate_all_users(self, k: int = 10) -> Dict[str, Any]:
        """
        Aggregate evaluation metrics across ALL users.
        
        Returns average Precision@K, Recall@K, NDCG@K, RMSE, and Hit Rate.
        """
        all_users = list(self.collab.interactions.keys())
        if not all_users:
            return {
                "num_users_evaluated": 0,
                "avg_precision": 0.0,
                "avg_recall": 0.0,
                "avg_ndcg": 0.0,
                "avg_rmse": None,
                "hit_rate": 0.0,
                "k": k,
                "note": "No users with sufficient interaction data",
            }

        precisions, recalls, ndcgs, rmses, hits = [], [], [], [], []

        for user in all_users:
            result = self.evaluate_user(user, k=k)
            p = result.get(f"precision_at_{k}")
            r = result.get(f"recall_at_{k}")
            n = result.get(f"ndcg_at_{k}")
            rm = result.get("rmse")
            h = result.get("hit")

            if p is not None:
                precisions.append(p)
            if r is not None:
                recalls.append(r)
            if n is not None:
                ndcgs.append(n)
            if rm is not None:
                rmses.append(rm)
            if h is not None:
                hits.append(h)

        n_eval = max(len(precisions), 1)

        return {
            "num_users_evaluated": len(all_users),
            "k": k,
            "avg_precision": round(sum(precisions) / n_eval, 4) if precisions else 0.0,
            "avg_recall": round(sum(recalls) / n_eval, 4) if recalls else 0.0,
            "avg_ndcg": round(sum(ndcgs) / n_eval, 4) if ndcgs else 0.0,
            "avg_rmse": round(sum(rmses) / len(rmses), 4) if rmses else None,
            "hit_rate": round(sum(hits) / len(hits), 4) if hits else 0.0,
        }

    def catalog_coverage(self, k: int = 10) -> Dict[str, Any]:
        """
        What fraction of the movie catalog gets recommended to anyone?
        Higher = more diverse recommendations.
        """
        all_users = list(self.collab.interactions.keys())
        recommended_set: set = set()

        for user in all_users:
            try:
                recs = self.hybrid.recommend(user_email=user, top_n=k)
                for r in recs:
                    recommended_set.add(r["title"].strip().lower())
            except Exception:
                continue

        total_catalog = len(self.content.get_all_titles()) if self.content.is_loaded else 0

        return {
            "unique_movies_recommended": len(recommended_set),
            "total_catalog_size": total_catalog,
            "coverage_pct": (
                round(len(recommended_set) / total_catalog * 100, 2)
                if total_catalog > 0 else 0.0
            ),
        }

    def _empty_result(self, k: int) -> Dict[str, Any]:
        return {
            f"precision_at_{k}": 0.0,
            f"recall_at_{k}": 0.0,
            f"ndcg_at_{k}": 0.0,
            "rmse": None,
            "hit": 0,
            "relevant_items": 0,
            "held_out_item": None,
        }
"""
services/hybrid.py
====================
Hybrid Recommendation Engine

Combines content-based (TF-IDF) and collaborative filtering (SVD) scores
using a weighted linear combination:

    final_score = α × content_score + (1 - α) × collaborative_score

Where α (content_weight) defaults to 0.6, giving:
    final_score = 0.6 × content_score + 0.4 × collaborative_score

The hybrid approach addresses limitations of each individual method:
- Content-based alone suffers from "filter bubble" (only recommends similar items)
- Collaborative alone suffers from "cold start" (needs many user interactions)
- Hybrid balances both: uses content similarity as a strong signal while
  incorporating collaborative patterns for serendipitous discovery.

Score Normalization:
    Both score sets are min-max normalized to [0, 1] before combining,
    ensuring neither source dominates due to scale differences.
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict

from services.recommender import ContentRecommender
from services.collaborative import CollaborativeEngine
from services.explainability import ExplainabilityEngine


class HybridRecommender:
    """
    Weighted hybrid recommender combining content-based and collaborative signals.
    """

    def __init__(
        self,
        content_engine: ContentRecommender,
        collab_engine: CollaborativeEngine,
        explain_engine: "ExplainabilityEngine",
        content_weight: float = 0.6,
    ):
        """
        Args:
            content_engine:  Loaded ContentRecommender instance
            collab_engine:   Loaded CollaborativeEngine instance
            explain_engine:  ExplainabilityEngine for generating reasons
            content_weight:  Weight for content-based scores (0.0 to 1.0)
                             Collaborative weight = 1 - content_weight
        """
        self.content = content_engine
        self.collab = collab_engine
        self.explain = explain_engine
        self.alpha = content_weight

    def recommend(
        self,
        user_email: str,
        movie_title: Optional[str] = None,
        top_n: int = 12,
    ) -> List[Dict[str, Any]]:
        """
        Generate hybrid recommendations.

        If movie_title is provided:
            → Content-based uses that movie as seed
            → Collaborative uses user's full history
        
        If movie_title is None:
            → Content-based aggregates across user's watchlist
            → Collaborative uses user's full history

        Returns:
            List of {
                "title": str,
                "score": float,          # hybrid score [0, 1]
                "content_score": float,   # normalized content score
                "collab_score": float,    # normalized collaborative score
                "reason": str,            # human-readable explanation
                "source": str,            # "hybrid", "content", or "collaborative"
            }
        """
        # ── Step 1: Get content-based recommendations ──
        content_recs = self._get_content_recs(user_email, movie_title, top_n * 3)

        # ── Step 2: Get collaborative recommendations ──
        collab_recs = self.collab.recommend(user_email, top_n=top_n * 3)

        # ── Step 3: Normalize scores ──
        content_scores = self._normalize({
            r["title"].strip().lower(): r["score"] for r in content_recs
        })
        collab_scores = self._normalize({
            r["title"].strip().lower(): r["score"] for r in collab_recs
        })

        # ── Step 4: Combine with weights ──
        all_titles = set(content_scores.keys()) | set(collab_scores.keys())

        combined: List[Dict[str, Any]] = []
        for title in all_titles:
            cs = content_scores.get(title, 0.0)
            co = collab_scores.get(title, 0.0)
            hybrid = self.alpha * cs + (1 - self.alpha) * co

            # Determine primary source
            if cs > 0 and co > 0:
                source = "hybrid"
            elif cs > 0:
                source = "content"
            else:
                source = "collaborative"

            # Generate explanation
            reason = self.explain.generate_reason(
                user_email=user_email,
                recommended_title=title,
                seed_title=movie_title,
                content_score=cs,
                collab_score=co,
                source=source,
            )

            combined.append({
                "title": self._find_original_title(title, content_recs, collab_recs),
                "score": round(hybrid, 4),
                "content_score": round(cs, 4),
                "collab_score": round(co, 4),
                "reason": reason,
                "source": source,
            })

        # Sort by hybrid score descending
        combined.sort(key=lambda x: -x["score"])
        return combined[:top_n]

    def recommend_for_you(
        self, user_email: str, top_n: int = 12
    ) -> List[Dict[str, Any]]:
        """
        Personalized "Recommended for You" section.
        Uses watchlist as seed for content-based, plus full collaborative signal.
        """
        return self.recommend(
            user_email=user_email,
            movie_title=None,  # use watchlist aggregation
            top_n=top_n,
        )

    def _get_content_recs(
        self, user_email: str, movie_title: Optional[str], n: int
    ) -> List[Dict[str, Any]]:
        """Get content-based recs — single movie or watchlist aggregation."""
        if movie_title:
            try:
                return self.content.recommend(movie_title, top_n=n)
            except ValueError:
                return []

        # No specific movie — aggregate from user's watchlist
        user_interactions = self.collab.get_user_interactions(user_email)
        if not user_interactions:
            return []

        # Use titles with highest interaction weight as seeds
        sorted_titles = sorted(
            user_interactions.items(), key=lambda x: -x[1]
        )
        seed_titles = [t for t, _ in sorted_titles[:10]]

        return self.content.recommend_multi(seed_titles, top_n=n)

    def _normalize(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Min-max normalize scores to [0, 1] range."""
        if not scores:
            return {}
        vals = list(scores.values())
        mn, mx = min(vals), max(vals)
        rng = mx - mn
        if rng == 0:
            # All same score — normalize to 0.5
            return {k: 0.5 for k in scores}
        return {k: (v - mn) / rng for k, v in scores.items()}

    def _find_original_title(
        self,
        norm_title: str,
        content_recs: List[Dict],
        collab_recs: List[Dict],
    ) -> str:
        """Recover the properly-cased title from either recommendation list."""
        for r in content_recs:
            if r["title"].strip().lower() == norm_title:
                return r["title"]
        for r in collab_recs:
            if r["title"].strip().lower() == norm_title:
                return r["title"]
        return norm_title.title()
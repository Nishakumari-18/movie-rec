"""
services/explainability.py
============================
Recommendation Explainability Engine

Generates human-readable explanations for each recommendation.
This is critical for:
1. User trust — people engage more with recommendations they understand
2. Academic value — demonstrates awareness of XAI (Explainable AI) principles
3. UX quality — "Because you watched X" is proven to increase click-through

Explanation templates vary based on the recommendation source:
- Content-based:   "Similar to {seed_movie}" / "Shares genres with your favorites"
- Collaborative:   "Users with similar taste also enjoyed this"
- Hybrid:          Combines both signals into a single explanation

The engine also uses the personalization data to reference specific
genres or movies the user has shown interest in.
"""

from typing import Optional, List, Dict, Any


class ExplainabilityEngine:
    """
    Generates contextual explanations for recommendations.
    
    Dependencies:
        - PersonalizationEngine (for user preference data)
        - ContentRecommender (for genre lookups)
    """

    def __init__(self, personalization_engine, content_engine):
        """
        Args:
            personalization_engine: services.personalization.PersonalizationEngine
            content_engine: services.recommender.ContentRecommender
        """
        self.personal = personalization_engine
        self.content = content_engine

    def generate_reason(
        self,
        user_email: str,
        recommended_title: str,
        seed_title: Optional[str] = None,
        content_score: float = 0.0,
        collab_score: float = 0.0,
        source: str = "hybrid",
    ) -> str:
        """
        Generate a human-readable explanation for why a movie was recommended.

        Args:
            user_email:        The user receiving the recommendation
            recommended_title: Title of the recommended movie
            seed_title:        The movie that triggered the recommendation (if any)
            content_score:     Normalized content-based score [0, 1]
            collab_score:      Normalized collaborative score [0, 1]
            source:            "content", "collaborative", or "hybrid"

        Returns:
            A natural language explanation string.
        """
        reasons = []

        # ── Content-based explanation ──
        if content_score > 0 and seed_title:
            reasons.append(f'Similar to "{seed_title.title()}"')

        if content_score > 0 and not seed_title:
            # Was recommended via watchlist aggregation
            liked = self.personal.get_liked_movies(user_email)
            if liked:
                # Find the closest liked movie (simple heuristic: use first match)
                closest = self._find_closest_liked(recommended_title, liked)
                if closest:
                    reasons.append(f'Because you watched "{closest.title()}"')
                else:
                    reasons.append("Matches your viewing history")

        # ── Genre-based explanation ──
        rec_genres = self.content.get_movie_genres(recommended_title)
        user_top_genres = self.personal.get_top_genres(user_email, top_n=3)
        user_genre_names = {g["genre"] for g in user_top_genres}

        matching_genres = [g for g in rec_genres if g in user_genre_names]
        if matching_genres:
            genre_str = " & ".join(matching_genres[:2])
            reasons.append(f"Matches your interest in {genre_str}")

        # ── Collaborative explanation ──
        if collab_score > 0.3:
            reasons.append("Users with similar taste enjoyed this")

        # ── Fallback ──
        if not reasons:
            if source == "content":
                reasons.append("Based on content similarity analysis")
            elif source == "collaborative":
                reasons.append("Recommended by our collaborative algorithm")
            else:
                reasons.append("Picked for you by our hybrid AI engine")

        # Combine (use top 2 reasons max for readability)
        return " · ".join(reasons[:2])

    def generate_section_explanation(
        self, section_type: str, user_email: str, seed_title: Optional[str] = None
    ) -> str:
        """
        Generate an explanation for an entire recommendation section.
        Used as subtitle text in the frontend.
        
        Args:
            section_type: "hybrid", "content", "collaborative", "for_you"
        """
        if section_type == "content" and seed_title:
            return f'Movies with similar themes, cast, and style to "{seed_title}"'

        if section_type == "collaborative":
            return "Based on viewing patterns of users with similar taste"

        if section_type == "hybrid" and seed_title:
            return f'AI-powered picks combining content analysis and user behavior for "{seed_title}"'

        if section_type == "for_you":
            top_genres = self.personal.get_top_genres(user_email, top_n=2)
            if top_genres:
                genre_str = ", ".join(g["genre"] for g in top_genres)
                return f"Personalized picks based on your love for {genre_str}"
            return "Personalized recommendations based on your activity"

        return "Smart recommendations powered by machine learning"

    def _find_closest_liked(
        self, rec_title: str, liked_titles: List[str]
    ) -> Optional[str]:
        """
        Find the liked movie most similar to the recommended one.
        Uses content similarity when available, falls back to genre overlap.
        """
        rec_norm = rec_title.strip().lower()
        rec_idx = self.content.get_index(rec_norm)

        if rec_idx is None:
            # Can't look up — return most recent liked
            return liked_titles[-1] if liked_titles else None

        best_title = None
        best_score = -1.0

        for liked in liked_titles:
            liked_idx = self.content.get_index(liked)
            if liked_idx is None:
                continue

            # Quick dot-product similarity between two specific movies
            try:
                score = float(
                    (self.content.tfidf_matrix[liked_idx] @
                     self.content.tfidf_matrix[rec_idx].T).toarray()[0, 0]
                )
                if score > best_score:
                    best_score = score
                    best_title = liked
            except Exception:
                continue

        return best_title

    def explain_algorithm(self) -> Dict[str, str]:
        """
        Return explanations of each algorithm for the Model Insights page.
        """
        return {
            "tfidf_content": (
                "TF-IDF Content-Based Filtering converts movie metadata "
                "(genres, keywords, cast, overview) into numerical vectors using "
                "Term Frequency-Inverse Document Frequency. Movies are recommended "
                "based on cosine similarity between their feature vectors. "
                "Strength: Works without user history. "
                "Weakness: Limited to item features, no personalization."
            ),
            "collaborative_svd": (
                "Collaborative Filtering with SVD builds a user-item interaction "
                "matrix from implicit feedback (watchlist=5, clicks=3, searches=1). "
                "Truncated SVD decomposes this matrix into latent factors that "
                "capture hidden patterns like 'likes dark thrillers' or "
                "'prefers feel-good comedies'. Predicted ratings for unseen movies "
                "are computed as U × Σ × Vᵀ. "
                "Strength: Discovers non-obvious preferences. "
                "Weakness: Cold-start problem for new users."
            ),
            "hybrid": (
                "The Hybrid Recommender combines both approaches: "
                "final_score = 0.6 × content_score + 0.4 × collab_score. "
                "Scores are min-max normalized before combining. This addresses "
                "the filter bubble of content-only systems and the cold-start "
                "of collaborative-only systems. The weight ratio can be tuned "
                "based on evaluation metrics."
            ),
            "explainability": (
                "Each recommendation includes a human-readable explanation "
                "generated from the recommendation source, user's genre preferences, "
                "and interaction history. This implements XAI (Explainable AI) "
                "principles — improving user trust and engagement."
            ),
        }
        
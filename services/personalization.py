"""
services/personalization.py
=============================
User Personalization & Behavior Tracking

Maintains a per-user preference profile that evolves over time:

    user_preferences = {
        "favorite_genres": {"Action": 12, "Drama": 8, ...},
        "liked_movies": ["inception", "the dark knight", ...],
        "interaction_count": {"watchlist": 5, "click": 23, "search": 41},
        "search_history": ["batman", "sci-fi", ...],
        "click_history": [{"tmdb_id": 123, "title": "...", "ts": "..."}, ...],
        "last_active": "2025-01-15T10:30:00",
    }

This data is used to:
1. Weight genre preferences for better recommendations
2. Power "Because you liked X" explanations
3. Build the "Recommended for You" feed
4. Provide analytics on the Model Insights page
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict


class PersonalizationEngine:
    """
    Tracks and manages user behavior profiles.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.prefs_path = os.path.join(data_dir, "user_preferences.json")
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        """Load preferences from disk."""
        if os.path.exists(self.prefs_path):
            try:
                with open(self.prefs_path, "r") as f:
                    self.profiles = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.profiles = {}
        print(f"[Personalization] Loaded profiles for {len(self.profiles)} users")

    def _save(self):
        """Persist preferences to disk."""
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self.prefs_path, "w") as f:
            json.dump(self.profiles, f, indent=2)

    def _ensure_profile(self, email: str) -> Dict[str, Any]:
        """Initialize a user profile if it doesn't exist."""
        if email not in self.profiles:
            self.profiles[email] = {
                "favorite_genres": {},
                "liked_movies": [],
                "interaction_count": {"watchlist": 0, "click": 0, "search": 0},
                "search_history": [],
                "click_history": [],
                "last_active": datetime.now().isoformat(),
            }
        return self.profiles[email]

    # ─── Record Events ───

    def record_click(
        self, email: str, tmdb_id: int, title: str, genres: Optional[List[str]] = None
    ):
        """Record when a user clicks/views a movie's detail page."""
        p = self._ensure_profile(email)
        p["interaction_count"]["click"] = p["interaction_count"].get("click", 0) + 1
        p["last_active"] = datetime.now().isoformat()

        # Add to click history (keep last 100)
        entry = {"tmdb_id": tmdb_id, "title": title, "ts": datetime.now().isoformat()}
        p["click_history"].append(entry)
        p["click_history"] = p["click_history"][-100:]

        # Update genre preferences
        if genres:
            for g in genres:
                p["favorite_genres"][g] = p["favorite_genres"].get(g, 0) + 1

        self._save()

    def record_watchlist(
        self, email: str, title: str, genres: Optional[List[str]] = None
    ):
        """Record when a user adds a movie to their watchlist."""
        p = self._ensure_profile(email)
        p["interaction_count"]["watchlist"] = p["interaction_count"].get("watchlist", 0) + 1
        p["last_active"] = datetime.now().isoformat()

        norm = title.strip().lower()
        if norm not in p["liked_movies"]:
            p["liked_movies"].append(norm)
            # Keep last 200
            p["liked_movies"] = p["liked_movies"][-200:]

        # Genre boost for watchlist (stronger signal)
        if genres:
            for g in genres:
                p["favorite_genres"][g] = p["favorite_genres"].get(g, 0) + 3

        self._save()

    def record_search(self, email: str, query: str):
        """Record a search query."""
        p = self._ensure_profile(email)
        p["interaction_count"]["search"] = p["interaction_count"].get("search", 0) + 1
        p["last_active"] = datetime.now().isoformat()

        p["search_history"].append(query.strip())
        # Deduplicate and keep last 50
        seen = set()
        unique = []
        for q in reversed(p["search_history"]):
            ql = q.lower()
            if ql not in seen:
                seen.add(ql)
                unique.append(q)
        p["search_history"] = list(reversed(unique[-50:]))

        self._save()

    # ─── Query Profiles ───

    def get_profile(self, email: str) -> Dict[str, Any]:
        """Get the full preference profile for a user."""
        return dict(self._ensure_profile(email))

    def get_top_genres(self, email: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Get user's top genres ranked by interaction frequency.
        Returns: [{"genre": "Action", "count": 12}, ...]
        """
        p = self._ensure_profile(email)
        genres = p.get("favorite_genres", {})
        sorted_genres = sorted(genres.items(), key=lambda x: -x[1])
        return [{"genre": g, "count": c} for g, c in sorted_genres[:top_n]]

    def get_recently_viewed(self, email: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's recent click history."""
        p = self._ensure_profile(email)
        history = p.get("click_history", [])
        return list(reversed(history[-limit:]))

    def get_liked_movies(self, email: str) -> List[str]:
        """Get list of movie titles the user has added to watchlist."""
        p = self._ensure_profile(email)
        return list(p.get("liked_movies", []))

    def get_search_history(self, email: str, limit: int = 10) -> List[str]:
        """Get recent search queries."""
        p = self._ensure_profile(email)
        return list(reversed(p.get("search_history", [])[-limit:]))

    def get_engagement_stats(self, email: str) -> Dict[str, Any]:
        """Get engagement statistics for a user (for Model Insights page)."""
        p = self._ensure_profile(email)
        counts = p.get("interaction_count", {})
        genres = p.get("favorite_genres", {})
        return {
            "total_clicks": counts.get("click", 0),
            "total_watchlist": counts.get("watchlist", 0),
            "total_searches": counts.get("search", 0),
            "total_interactions": sum(counts.values()),
            "genres_explored": len(genres),
            "movies_liked": len(p.get("liked_movies", [])),
            "last_active": p.get("last_active", "Never"),
        }

    # ─── Global Stats ───

    def global_stats(self) -> Dict[str, Any]:
        """Aggregate stats across all users (for admin / metrics page)."""
        total_users = len(self.profiles)
        total_interactions = 0
        genre_global: Dict[str, int] = defaultdict(int)

        for email, p in self.profiles.items():
            counts = p.get("interaction_count", {})
            total_interactions += sum(counts.values())
            for g, c in p.get("favorite_genres", {}).items():
                genre_global[g] += c

        top_genres = sorted(genre_global.items(), key=lambda x: -x[1])[:10]

        return {
            "total_users": total_users,
            "total_interactions": total_interactions,
            "avg_interactions_per_user": (
                round(total_interactions / total_users, 1) if total_users > 0 else 0
            ),
            "top_genres_global": [{"genre": g, "count": c} for g, c in top_genres],
        }
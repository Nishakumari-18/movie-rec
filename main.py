"""
main.py
=========
FastAPI Backend — Movie Hybrid Recommendation System

Architecture:
    ┌─────────────┐     ┌──────────────────┐
    │  Streamlit   │────▶│    FastAPI API    │
    │  Frontend    │◀────│   (this file)    │
    └─────────────┘     └──────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ Content-Based │  │Collaborative │  │    Hybrid     │
    │   TF-IDF     │  │     SVD      │  │   Combiner    │
    └──────────────┘  └──────────────┘  └──────────────┘
              │                │                 │
              └────────────────┼─────────────────┘
                               ▼
                    ┌──────────────────┐
                    │ Personalization  │
                    │ Explainability   │
                    │    Metrics       │
                    └──────────────────┘

All existing endpoints are preserved. New endpoints are added for:
- Collaborative filtering recommendations
- Hybrid recommendations
- User interaction tracking
- Personalization profiles
- Evaluation metrics
"""

import os
from typing import Optional, List, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ── Import service modules ──
from services.recommender import ContentRecommender
from services.collaborative import CollaborativeEngine
from services.personalization import PersonalizationEngine
from services.explainability import ExplainabilityEngine
from services.hybrid import HybridRecommender
from services.metrics import MetricsEngine

# =========================
# ENV
# =========================
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY missing. Put it in .env as TMDB_API_KEY=xxxx")


# =========================
# FASTAPI APP
# =========================
app = FastAPI(
    title="Movie Hybrid Recommendation System",
    version="4.0",
    description="Content-based + Collaborative + Hybrid recommendation engine",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# SERVICE INSTANCES (globals)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

content_engine = ContentRecommender(BASE_DIR)
collab_engine = CollaborativeEngine(DATA_DIR, n_factors=50)
personal_engine = PersonalizationEngine(DATA_DIR)

# These depend on the above, so initialized after startup
explain_engine: Optional[ExplainabilityEngine] = None
hybrid_engine: Optional[HybridRecommender] = None
metrics_engine: Optional[MetricsEngine] = None


# =========================
# PYDANTIC MODELS
# =========================
class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class HybridRecItem(BaseModel):
    title: str
    score: float
    content_score: float
    collab_score: float
    reason: str
    source: str
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]


class InteractionRequest(BaseModel):
    user_email: str
    movie_title: str
    interaction_type: str  # "watchlist", "click", "search"
    tmdb_id: Optional[int] = None
    genres: Optional[List[str]] = None


# =========================
# TMDB UTILITIES (preserved from original)
# =========================
def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"


async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    q = dict(params)
    q["api_key"] = TMDB_API_KEY
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{TMDB_BASE}{path}", params=q)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"TMDB request error: {repr(e)}")
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"TMDB error {r.status_code}: {r.text}")
    return r.json()


async def tmdb_cards_from_results(results: List[dict], limit: int = 20) -> List[TMDBMovieCard]:
    out = []
    for m in (results or [])[:limit]:
        out.append(TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or m.get("name") or "",
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        ))
    return out


async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
    )


async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    return await tmdb_get(
        "/search/movie",
        {"query": query, "include_adult": "false", "language": "en-US", "page": page},
    )


async def tmdb_search_first(query: str) -> Optional[dict]:
    data = await tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None


async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    try:
        m = await tmdb_search_first(title)
        if not m:
            return None
        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
    except Exception:
        return None


# =========================
# STARTUP
# =========================
@app.on_event("startup")
def startup():
    global explain_engine, hybrid_engine, metrics_engine

    # Load content-based engine (pickles)
    content_engine.load()

    # Load collaborative engine (user interactions)
    collab_engine.load_interactions()

    # Initialize dependent engines
    explain_engine = ExplainabilityEngine(personal_engine, content_engine)
    hybrid_engine = HybridRecommender(
        content_engine=content_engine,
        collab_engine=collab_engine,
        explain_engine=explain_engine,
        content_weight=0.6,
    )
    metrics_engine = MetricsEngine(collab_engine, content_engine, hybrid_engine)

    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("  Movie Hybrid Recommendation System v4.0")
    print(f"  Content engine: {len(content_engine.title_to_idx)} movies loaded")
    print(f"  Collab engine:  {collab_engine.stats['total_users']} users tracked")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════
# EXISTING ENDPOINTS (preserved from original)
# ═══════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "4.0",
        "content_movies": len(content_engine.title_to_idx) if content_engine.is_loaded else 0,
        "collab_users": collab_engine.stats["total_users"],
    }


@app.get("/home", response_model=List[TMDBMovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
):
    """Home feed (TMDB trending/popular/etc.)"""
    try:
        if category == "trending":
            data = await tmdb_get("/trending/movie/day", {"language": "en-US"})
            return await tmdb_cards_from_results(data.get("results", []), limit=limit)
        if category not in {"popular", "top_rated", "upcoming", "now_playing"}:
            raise HTTPException(status_code=400, detail="Invalid category")
        data = await tmdb_get(f"/movie/{category}", {"language": "en-US", "page": 1})
        return await tmdb_cards_from_results(data.get("results", []), limit=limit)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Home route failed: {e}")


@app.get("/tmdb/search")
async def tmdb_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=10),
):
    """Raw TMDB keyword search."""
    return await tmdb_search_movies(query=query, page=page)


@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_route(tmdb_id: int):
    return await tmdb_movie_details(tmdb_id)


@app.get("/recommend/genre", response_model=List[TMDBMovieCard])
async def recommend_genre(
    tmdb_id: int = Query(...),
    limit: int = Query(18, ge=1, le=50),
):
    """Genre-based recommendations via TMDB discover."""
    details = await tmdb_movie_details(tmdb_id)
    if not details.genres:
        return []
    genre_id = details.genres[0]["id"]
    discover = await tmdb_get(
        "/discover/movie",
        {"with_genres": genre_id, "language": "en-US", "sort_by": "popularity.desc", "page": 1},
    )
    cards = await tmdb_cards_from_results(discover.get("results", []), limit=limit)
    return [c for c in cards if c.tmdb_id != tmdb_id]


@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    """Content-based TF-IDF recommendations."""
    try:
        recs = content_engine.recommend(title, top_n=top_n)
        return [{"title": r["title"], "score": r["score"]} for r in recs]
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    """Bundle: movie details + TF-IDF recs + genre recs."""
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(status_code=404, detail=f"No TMDB movie found for: {query}")

    tmdb_id = int(best["id"])
    details = await tmdb_movie_details(tmdb_id)

    # TF-IDF recommendations
    tfidf_items = []
    try:
        recs = content_engine.recommend(details.title, top_n=tfidf_top_n)
    except Exception:
        try:
            recs = content_engine.recommend(query, top_n=tfidf_top_n)
        except Exception:
            recs = []

    for r in recs:
        card = await attach_tmdb_card_by_title(r["title"])
        tfidf_items.append(TFIDFRecItem(title=r["title"], score=r["score"], tmdb=card))

    # Genre recommendations
    genre_recs = []
    if details.genres:
        genre_id = details.genres[0]["id"]
        discover = await tmdb_get(
            "/discover/movie",
            {"with_genres": genre_id, "language": "en-US", "sort_by": "popularity.desc", "page": 1},
        )
        cards = await tmdb_cards_from_results(discover.get("results", []), limit=genre_limit)
        genre_recs = [c for c in cards if c.tmdb_id != details.tmdb_id]

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )


# ═══════════════════════════════════════════════════════════
# NEW ENDPOINTS: Collaborative Filtering
# ═══════════════════════════════════════════════════════════

@app.get("/recommend/collaborative")
async def recommend_collaborative(
    user_email: str = Query(..., min_length=1),
    top_n: int = Query(12, ge=1, le=50),
):
    """
    Collaborative filtering recommendations based on user behavior.
    Uses SVD matrix factorization on the user-item interaction matrix.
    """
    recs = collab_engine.recommend(user_email, top_n=top_n)

    # Attach TMDB posters
    results = []
    for r in recs:
        card = await attach_tmdb_card_by_title(r["title"])
        results.append({
            "title": r["title"],
            "score": round(r["score"], 4),
            "tmdb": card.dict() if card else None,
        })
    return results


# ═══════════════════════════════════════════════════════════
# NEW ENDPOINTS: Hybrid Recommendations
# ═══════════════════════════════════════════════════════════

@app.get("/recommend/hybrid")
async def recommend_hybrid(
    user_email: str = Query(..., min_length=1),
    movie_title: Optional[str] = Query(None),
    top_n: int = Query(12, ge=1, le=30),
):
    """
    Hybrid recommendations combining content-based and collaborative signals.
    
    final_score = 0.6 × content_score + 0.4 × collab_score
    
    If movie_title is provided, recommendations are seeded from that movie.
    Otherwise, recommendations are based on the user's full interaction history.
    """
    if hybrid_engine is None:
        raise HTTPException(status_code=500, detail="Hybrid engine not initialized")

    recs = hybrid_engine.recommend(
        user_email=user_email,
        movie_title=movie_title,
        top_n=top_n,
    )

    # Attach TMDB poster data
    results = []
    for r in recs:
        card = await attach_tmdb_card_by_title(r["title"])
        results.append(HybridRecItem(
            title=r["title"],
            score=r["score"],
            content_score=r["content_score"],
            collab_score=r["collab_score"],
            reason=r["reason"],
            source=r["source"],
            tmdb=card,
        ))
    return results


@app.get("/recommend/for-you")
async def recommend_for_you(
    user_email: str = Query(..., min_length=1),
    top_n: int = Query(12, ge=1, le=30),
):
    """
    Personalized "Recommended For You" endpoint.
    Combines watchlist-based content analysis with collaborative patterns.
    """
    if hybrid_engine is None:
        raise HTTPException(status_code=500, detail="Hybrid engine not initialized")

    recs = hybrid_engine.recommend_for_you(user_email=user_email, top_n=top_n)

    results = []
    for r in recs:
        card = await attach_tmdb_card_by_title(r["title"])
        results.append(HybridRecItem(
            title=r["title"],
            score=r["score"],
            content_score=r["content_score"],
            collab_score=r["collab_score"],
            reason=r["reason"],
            source=r["source"],
            tmdb=card,
        ))
    return results


# ═══════════════════════════════════════════════════════════
# NEW ENDPOINTS: Interaction Tracking
# ═══════════════════════════════════════════════════════════

@app.post("/track/interaction")
async def track_interaction(req: InteractionRequest):
    """
    Record a user-movie interaction for collaborative filtering.
    
    interaction_type: "watchlist" | "click" | "search"
    This updates both the collaborative engine and personalization profile.
    """
    # Update collaborative engine
    collab_engine.record_interaction(
        req.user_email, req.movie_title, req.interaction_type
    )

    # Update personalization profile
    if req.interaction_type == "click":
        personal_engine.record_click(
            req.user_email,
            tmdb_id=req.tmdb_id or 0,
            title=req.movie_title,
            genres=req.genres,
        )
    elif req.interaction_type == "watchlist":
        personal_engine.record_watchlist(
            req.user_email,
            title=req.movie_title,
            genres=req.genres,
        )
    elif req.interaction_type == "search":
        personal_engine.record_search(req.user_email, req.movie_title)

    return {"status": "recorded", "type": req.interaction_type}


@app.post("/track/sync-watchlist")
async def sync_watchlist(user_email: str, titles: List[str]):
    """Bulk sync watchlist into collaborative engine."""
    collab_engine.record_watchlist_bulk(user_email, titles)
    return {"status": "synced", "count": len(titles)}


# ═══════════════════════════════════════════════════════════
# NEW ENDPOINTS: Personalization
# ═══════════════════════════════════════════════════════════

@app.get("/user/profile")
async def user_profile(user_email: str = Query(..., min_length=1)):
    """Get the user's personalization profile."""
    return personal_engine.get_profile(user_email)


@app.get("/user/top-genres")
async def user_top_genres(
    user_email: str = Query(..., min_length=1),
    top_n: int = Query(5, ge=1, le=20),
):
    """Get user's most-interacted genres."""
    return personal_engine.get_top_genres(user_email, top_n=top_n)


@app.get("/user/stats")
async def user_stats(user_email: str = Query(..., min_length=1)):
    """Get engagement statistics for a user."""
    return personal_engine.get_engagement_stats(user_email)


# ═══════════════════════════════════════════════════════════
# NEW ENDPOINTS: Metrics & Evaluation
# ═══════════════════════════════════════════════════════════

@app.get("/metrics")
async def get_metrics(k: int = Query(10, ge=1, le=50)):
    """
    Evaluate the recommendation system.
    
    Returns: Precision@K, Recall@K, NDCG@K, RMSE, Hit Rate
    averaged across all users.
    """
    if metrics_engine is None:
        raise HTTPException(status_code=500, detail="Metrics engine not initialized")

    results = metrics_engine.evaluate_all_users(k=k)
    coverage = metrics_engine.catalog_coverage(k=k)

    return {
        **results,
        **coverage,
        "collab_stats": collab_engine.stats,
        "personalization_stats": personal_engine.global_stats(),
    }


@app.get("/metrics/user")
async def get_user_metrics(
    user_email: str = Query(..., min_length=1),
    k: int = Query(10, ge=1, le=50),
):
    """Evaluate recommendation quality for a specific user."""
    if metrics_engine is None:
        raise HTTPException(status_code=500, detail="Metrics engine not initialized")
    return metrics_engine.evaluate_user(user_email, k=k)


@app.get("/metrics/algorithms")
async def algorithm_explanations():
    """Get explanations of each algorithm (for Model Insights page)."""
    if explain_engine is None:
        raise HTTPException(status_code=500, detail="Explain engine not initialized")
    return explain_engine.explain_algorithm()
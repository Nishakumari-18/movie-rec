"""
app.py — Streamlit Frontend for Hybrid Movie Recommendation System

Upgraded UI with:
1. "Recommended for You"    → /recommend/hybrid (personalized)
2. "Because You Watched"    → shows explanation per recommendation
3. "Similar Movies"         → /recommend/tfidf (content-based)
4. "Based on Your Watchlist"→ /recommend/collaborative
5. "Model Insights" page    → /metrics + algorithm explanations
"""

import requests
import streamlit as st
import json
import os
import hashlib
from datetime import datetime

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
API_BASE = os.getenv("API_BASE", "https://movie-rec-466x.onrender.com") or "http://127.0.0.1:8000"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"
USERS_FILE = "users_db.json"
WATCHLIST_FILE = "watchlist_db.json"
SESSION_FILE = "session_db.json"

st.set_page_config(page_title="Movie AI — Hybrid Recommender", page_icon="🎬", layout="wide")


# ═══════════════════════════════════════════════════════════
# DATABASE HELPERS (preserved)
# ═══════════════════════════════════════════════════════════

def _read_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}

def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def _hash(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def signup_user(name, email, pw):
    users = _read_json(USERS_FILE)
    if email in users:
        return False, "Email already registered."
    users[email] = {"name": name, "email": email, "password": _hash(pw), "created_at": datetime.now().isoformat()}
    _write_json(USERS_FILE, users)
    return True, "Account created!"

def login_user(email, pw):
    users = _read_json(USERS_FILE)
    if email not in users:
        return False, "Email not found."
    if users[email]["password"] != _hash(pw):
        return False, "Incorrect password."
    return True, users[email]

def save_session(email, name):
    _write_json(SESSION_FILE, {"email": email, "name": name, "ts": datetime.now().isoformat()})

def load_session():
    data = _read_json(SESSION_FILE)
    if data.get("email"):
        users = _read_json(USERS_FILE)
        if data["email"] in users:
            return data
    return None

def clear_session():
    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)

def get_watchlist(email):
    data = _read_json(WATCHLIST_FILE)
    return data.get(email, [])

def add_to_watchlist(email, movie):
    data = _read_json(WATCHLIST_FILE)
    if email not in data:
        data[email] = []
    tid = int(movie.get("tmdb_id", 0))
    for m in data[email]:
        if int(m.get("tmdb_id", 0)) == tid:
            return False
    data[email].append({
        "tmdb_id": tid,
        "title": movie.get("title", ""),
        "poster_url": movie.get("poster_url"),
        "added_at": datetime.now().isoformat(),
    })
    _write_json(WATCHLIST_FILE, data)

    # ── NEW: Track in backend for collaborative filtering ──
    _track_interaction(email, movie.get("title", ""), "watchlist", tmdb_id=tid)
    return True

def remove_from_watchlist(email, tmdb_id):
    data = _read_json(WATCHLIST_FILE)
    if email not in data:
        return False
    tid = int(tmdb_id)
    before = len(data[email])
    data[email] = [m for m in data[email] if int(m.get("tmdb_id", 0)) != tid]
    _write_json(WATCHLIST_FILE, data)
    return len(data[email]) < before

def is_in_watchlist(email, tmdb_id):
    for m in get_watchlist(email):
        if int(m.get("tmdb_id", 0)) == int(tmdb_id):
            return True
    return False


# ═══════════════════════════════════════════════════════════
# NEW: Interaction Tracking Helper
# ═══════════════════════════════════════════════════════════

def _track_interaction(email, title, itype, tmdb_id=None, genres=None):
    """Send interaction event to backend for collaborative filtering."""
    try:
        requests.post(
            f"{API_BASE}/track/interaction",
            json={
                "user_email": email,
                "movie_title": title,
                "interaction_type": itype,
                "tmdb_id": tmdb_id,
                "genres": genres,
            },
            timeout=5,
        )
    except Exception:
        pass  # Non-critical; don't break UI


# ═══════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════

DEFAULTS = {
    "view": "home",
    "selected_tmdb_id": None,
    "logged_in": False,
    "user_email": None,
    "user_name": None,
    "auth_page": "login",
    "category": "trending",
    "recent_searches": [],
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if not st.session_state.logged_in:
    saved = load_session()
    if saved:
        st.session_state.logged_in = True
        st.session_state.user_email = saved["email"]
        st.session_state.user_name = saved["name"]

qp_v = st.query_params.get("view")
qp_id = st.query_params.get("id")
if qp_v in ("home", "details", "watchlist", "insights"):
    st.session_state.view = qp_v
if qp_id:
    try:
        st.session_state.selected_tmdb_id = int(qp_id)
        st.session_state.view = "details"
    except ValueError:
        pass


# ═══════════════════════════════════════════════════════════
# NAVIGATION
# ═══════════════════════════════════════════════════════════

def goto_home():
    st.session_state.view = "home"
    st.session_state.selected_tmdb_id = None
    st.query_params.clear()
    st.query_params["view"] = "home"
    st.rerun()

def goto_details(tmdb_id):
    st.session_state.view = "details"
    st.session_state.selected_tmdb_id = int(tmdb_id)
    st.query_params.clear()
    st.query_params["view"] = "details"
    st.query_params["id"] = str(int(tmdb_id))
    st.rerun()

def goto_watchlist():
    st.session_state.view = "watchlist"
    st.session_state.selected_tmdb_id = None
    st.query_params.clear()
    st.query_params["view"] = "watchlist"
    st.rerun()

def goto_insights():
    st.session_state.view = "insights"
    st.session_state.selected_tmdb_id = None
    st.query_params.clear()
    st.query_params["view"] = "insights"
    st.rerun()

def do_logout():
    clear_session()
    for k in DEFAULTS:
        st.session_state[k] = DEFAULTS[k]
    st.query_params.clear()
    st.rerun()


# ═══════════════════════════════════════════════════════════
# API HELPERS
# ═══════════════════════════════════════════════════════════

@st.cache_data(ttl=30)
def api_get_json(path, params=None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=25)
        if r.status_code >= 400:
            return None, f"HTTP {r.status_code}"
        return r.json(), None
    except Exception as e:
        return None, str(e)

def api_post_json(path, body=None):
    try:
        r = requests.post(f"{API_BASE}{path}", json=body, timeout=10)
        return r.json(), None
    except Exception as e:
        return None, str(e)

def safe_image(url):
    try:
        st.image(url, use_container_width=True)
    except TypeError:
        try:
            st.image(url, use_column_width=True)
        except Exception:
            st.image(url)


# ═══════════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════════

def render_poster_grid(cards, cols=6, kp="g", show_wl=False, show_reason=False):
    """Render a grid of movie posters with optional reason badges."""
    if not cards:
        st.info("No movies to show.")
        return
    email = st.session_state.get("user_email")
    total = len(cards)
    rows = (total + cols - 1) // cols
    idx = 0
    for r in range(rows):
        columns = st.columns(cols)
        for c in range(cols):
            if idx >= total:
                break
            m = cards[idx]
            idx += 1

            # Support both dict shapes (flat and nested tmdb)
            tmdb = m.get("tmdb") or {}
            tid = m.get("tmdb_id") or tmdb.get("tmdb_id")
            title = m.get("title") or tmdb.get("title") or "Untitled"
            poster = m.get("poster_url") or tmdb.get("poster_url")
            reason = m.get("reason", "")

            with columns[c]:
                if poster:
                    safe_image(poster)
                else:
                    short = title[:18] + ("…" if len(title) > 18 else "")
                    st.markdown(f"<div class='no-p'><div class='no-p-i'>🎬</div><span>{short}</span></div>", unsafe_allow_html=True)

                if st.button("▶ View", key=f"v_{kp}_{tid}_{idx}"):
                    if tid:
                        # Track click interaction
                        if email:
                            _track_interaction(email, title, "click", tmdb_id=tid)
                        goto_details(tid)

                if show_wl and email and tid:
                    if is_in_watchlist(email, tid):
                        st.markdown("<div class='wl-r'>", unsafe_allow_html=True)
                        if st.button("✕ Remove", key=f"wr_{kp}_{tid}_{idx}"):
                            remove_from_watchlist(email, int(tid))
                            st.rerun()
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='wl-a'>", unsafe_allow_html=True)
                        if st.button("♡ Watchlist", key=f"wa_{kp}_{tid}_{idx}"):
                            add_to_watchlist(email, {"tmdb_id": int(tid), "title": title, "poster_url": poster})
                            st.rerun()
                        st.markdown("</div>", unsafe_allow_html=True)

                st.markdown(f"<div class='movie-card-title'>{title}</div>", unsafe_allow_html=True)

                # Show recommendation reason if available
                if show_reason and reason:
                    st.markdown(
                        f"<div style='font-size:0.68rem;color:var(--t3);padding:2px 4px;"
                        f"background:rgba(229,9,20,0.05);border-radius:6px;margin-top:2px;"
                        f"line-height:1.3'>💡 {reason}</div>",
                        unsafe_allow_html=True,
                    )


def to_tfidf_cards(items):
    out = []
    for x in items or []:
        t = x.get("tmdb") or {}
        if t.get("tmdb_id"):
            out.append({"tmdb_id": t["tmdb_id"], "title": t.get("title") or x.get("title", ""), "poster_url": t.get("poster_url")})
    return out


def to_hybrid_cards(items):
    """Convert hybrid API response to card format, preserving reasons."""
    out = []
    for x in items or []:
        t = x.get("tmdb") or {}
        card = {
            "title": x.get("title") or t.get("title", ""),
            "tmdb_id": t.get("tmdb_id"),
            "poster_url": t.get("poster_url"),
            "reason": x.get("reason", ""),
            "score": x.get("score", 0),
            "source": x.get("source", ""),
        }
        if card["tmdb_id"]:
            out.append(card)
    return out


def parse_search(data, kw, limit=24):
    kwl = kw.strip().lower()
    raw = []
    if isinstance(data, dict) and "results" in data:
        for m in data.get("results") or []:
            t = (m.get("title") or "").strip()
            tid = m.get("id")
            pp = m.get("poster_path")
            if not t or not tid:
                continue
            raw.append({"tmdb_id": int(tid), "title": t, "poster_url": f"{TMDB_IMG}{pp}" if pp else None, "release_date": m.get("release_date", "")})
    elif isinstance(data, list):
        for m in data:
            tid = m.get("tmdb_id") or m.get("id")
            t = (m.get("title") or "").strip()
            if not t or not tid:
                continue
            raw.append({"tmdb_id": int(tid), "title": t, "poster_url": m.get("poster_url"), "release_date": m.get("release_date", "")})
    else:
        return [], []

    matched = [x for x in raw if kwl in x["title"].lower()]
    final = matched if matched else raw
    suggestions = []
    for x in final[:10]:
        yr = (x.get("release_date") or "")[:4]
        suggestions.append((f"{x['title']} ({yr})" if yr else x["title"], x["tmdb_id"]))
    cards = [{"tmdb_id": x["tmdb_id"], "title": x["title"], "poster_url": x["poster_url"]} for x in final[:limit]]
    return suggestions, cards


CATS = {
    "trending": ("Trending", "🔥"),
    "popular": ("Popular", "⭐"),
    "top_rated": ("Top Rated", "🏆"),
    "now_playing": ("Now Playing", "🎬"),
    "upcoming": ("Upcoming", "📅"),
}


# ═══════════════════════════════════════════════════════════
# CSS (all your existing CSS preserved — abbreviated here for space)
# ═══════════════════════════════════════════════════════════

BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
:root {
    --bg: #06080f; --bg2: #0b0f1a;
    --surface: rgba(15,23,42,0.55); --surface-hi: rgba(15,23,42,0.85);
    --border: rgba(255,255,255,0.06); --border-hi: rgba(255,255,255,0.12);
    --red: #e50914; --red-lt: #ff1a25; --red-glow: rgba(229,9,20,0.30);
    --red-grad: linear-gradient(135deg,#e50914,#b20710);
    --green: #059669;
    --t1: #f1f5f9; --t2: #94a3b8; --t3: #64748b;
    --radius-sm: 10px; --radius-md: 14px; --radius-lg: 20px; --radius-pill: 999px;
    --ease: cubic-bezier(0.4,0,0.2,1);
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"],
.main, .block-container, [data-testid="stMainBlockContainer"] {
    background: var(--bg) !important; color: var(--t1) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
#MainMenu, header[data-testid="stHeader"], footer,
[data-testid="stToolbar"], .stDeployButton, [data-testid="stDecoration"] { display: none !important; }
*, h1, h2, h3, h4, h5, h6, p, span, div, label, li { font-family: 'Inter', -apple-system, sans-serif !important; }
h1 { display: none !important; }
h3, h4 { color: var(--t1) !important; font-weight: 700 !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 10px; }
.nav { background: rgba(6,8,15,0.85); backdrop-filter: blur(20px) saturate(1.4); border-bottom: 1px solid var(--border); padding: 12px 0; margin: 0 -1rem; position: sticky; top: 0; z-index: 999; }
.nav-in { max-width: 1400px; margin: 0 auto; display: flex; align-items: center; justify-content: space-between; padding: 0 2rem; }
.brand { display: flex; align-items: center; gap: 12px; }
.brand-i { background: var(--red-grad); width: 36px; height: 36px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; box-shadow: 0 0 20px var(--red-glow); }
.brand-t { font-weight: 800; font-size: 1.35rem; letter-spacing: -0.5px; background: linear-gradient(135deg, #fff 30%, #94a3b8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.nav-rt { display: flex; align-items: center; gap: 10px; }
.nav-nm { font-size: 0.85rem; font-weight: 600; color: var(--t1); }
.nav-av { width: 34px; height: 34px; border-radius: 50%; background: linear-gradient(135deg, #e50914, #7c3aed); display: flex; align-items: center; justify-content: center; font-size: 0.85rem; font-weight: 700; color: #fff; }
@keyframes fadeInUp { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: none; } }
.fi { animation: fadeInUp 0.45s var(--ease) both; }
.fi1 { animation-delay: 0.08s; } .fi2 { animation-delay: 0.16s; } .fi3 { animation-delay: 0.24s; }
[data-testid="InputInstructions"] { display: none !important; }
.sh { display: flex; align-items: center; gap: 10px; margin: 1.8rem 0 1rem; padding-bottom: 10px; border-bottom: 1px solid var(--border); }
.sh .e { font-size: 1.2rem; } .sh .l { font-size: 1.1rem; font-weight: 700; color: var(--t1); }
.sh .b { background: rgba(229,9,20,0.08); color: var(--red); font-size: 0.65rem; font-weight: 700; padding: 3px 10px; border-radius: var(--radius-pill); letter-spacing: 0.8px; text-transform: uppercase; border: 1px solid rgba(229,9,20,0.12); }
.movie-card-title { font-size: 0.78rem; font-weight: 600; color: var(--t2); line-height: 1.35; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; min-height: 2.1rem; padding: 6px 2px 2px; }
[data-testid="stImage"] { border-radius: var(--radius-md) !important; overflow: hidden !important; box-shadow: 0 2px 12px rgba(0,0,0,0.25) !important; transition: all 0.3s var(--ease) !important; cursor: pointer !important; }
[data-testid="stImage"]:hover { box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 20px rgba(229,9,20,0.08) !important; transform: translateY(-4px) !important; }
.no-p { height: 280px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; background: linear-gradient(145deg, #111827, #0a0e1a); border-radius: var(--radius-md); border: 1px solid var(--border); color: var(--t3); font-size: 0.78rem; }
.no-p-i { font-size: 2.4rem; opacity: 0.35; }
.dc { background: var(--surface); backdrop-filter: blur(20px); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 28px; box-shadow: 0 4px 16px rgba(0,0,0,0.2); }
.dt { font-size: 1.9rem; font-weight: 800; letter-spacing: -0.8px; line-height: 1.15; margin-bottom: 14px; background: linear-gradient(135deg, #fff, #cbd5e1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.dm { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }
.chip { display: inline-flex; align-items: center; gap: 5px; background: rgba(255,255,255,0.04); border: 1px solid var(--border); border-radius: var(--radius-pill); padding: 5px 14px; font-size: 0.74rem; font-weight: 500; color: var(--t2); }
.chip-a { background: rgba(229,9,20,0.08); border-color: rgba(229,9,20,0.15); color: var(--red-lt); font-weight: 600; }
.ovl { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; color: var(--red); margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }
.ovl::after { content: ''; flex: 1; height: 1px; background: var(--border); }
.ovt { color: var(--t2); font-size: 0.92rem; line-height: 1.8; }
.dh { position: relative; border-radius: var(--radius-lg); overflow: hidden; margin-bottom: 1.5rem; box-shadow: 0 12px 40px rgba(0,0,0,0.5); }
.dh img { width: 100%; height: 420px; object-fit: cover; filter: brightness(0.3) saturate(1.3); display: block; }
.dh-ov { position: absolute; bottom: 0; left: 0; right: 0; height: 70%; background: linear-gradient(to top, var(--bg), transparent); }
.dh-s { position: absolute; left: 0; top: 0; bottom: 0; width: 25%; background: linear-gradient(to right, var(--bg), transparent); }
.pw { border-radius: var(--radius-lg); overflow: hidden; box-shadow: 0 16px 48px rgba(0,0,0,0.4); border: 1px solid var(--border); }
.stat { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 20px 24px; text-align: center; }
.stat-n { font-size: 1.8rem; font-weight: 800; background: linear-gradient(135deg, #fff, #e50914); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.stat-l { font-size: 0.7rem; color: var(--t3); text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-top: 4px; }
.empty { text-align: center; padding: 60px 20px; }
.empty-i { font-size: 3.2rem; margin-bottom: 14px; opacity: 0.3; }
.empty-h { font-size: 1.15rem; font-weight: 700; color: var(--t2); margin-bottom: 6px; }
.empty-p { font-size: 0.86rem; color: var(--t3); }
/* Metric card for insights */
.metric-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 24px; margin-bottom: 12px; }
.metric-title { font-size: 0.72rem; color: var(--t3); text-transform: uppercase; letter-spacing: 1.2px; font-weight: 600; margin-bottom: 6px; }
.metric-val { font-size: 2rem; font-weight: 800; color: var(--t1); }
.metric-sub { font-size: 0.78rem; color: var(--t3); margin-top: 4px; }
.algo-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 24px; margin-bottom: 16px; }
.algo-name { font-size: 1rem; font-weight: 700; color: var(--red-lt); margin-bottom: 8px; }
.algo-desc { font-size: 0.85rem; color: var(--t2); line-height: 1.7; }
</style>
"""

# Auth CSS (abbreviated — same as original)
AUTH_CSS = """
<style>
.block-container, [data-testid="stMainBlockContainer"] { padding: 0 1rem !important; max-width: 100% !important; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], button[kind="header"] { display: none !important; }
.auth-page { display: flex; align-items: center; justify-content: center; padding: 20px 16px; }
.auth-card { width: 100%; max-width: 400px; background: rgba(15,23,42,0.5); backdrop-filter: blur(40px) saturate(1.5); border: 1px solid rgba(255,255,255,0.07); border-radius: 24px; padding: 40px 32px; box-shadow: 0 24px 64px rgba(0,0,0,0.5); animation: authSlideUp 0.5s var(--ease) both; }
@keyframes authSlideUp { from { opacity: 0; transform: translateY(24px) scale(0.97); } to { opacity: 1; transform: none; } }
.auth-logo { text-align: center; margin-bottom: 20px; }
.auth-ico { display: inline-flex; align-items: center; justify-content: center; width: 56px; height: 56px; border-radius: 16px; background: var(--red-grad); font-size: 1.5rem; box-shadow: 0 0 40px var(--red-glow); }
.auth-h { text-align: center; font-size: 1.4rem; font-weight: 800; color: var(--t1); margin-bottom: 4px; }
.auth-sub { text-align: center; color: var(--t3); font-size: 0.82rem; margin-bottom: 20px; line-height: 1.5; }
.auth-or { display: flex; align-items: center; gap: 14px; margin: 16px 0; color: var(--t3); font-size: 0.74rem; }
.auth-or::before, .auth-or::after { content: ''; flex: 1; height: 1px; background: var(--border-hi); }
.t-ok { background: rgba(5,150,105,0.1); border: 1px solid rgba(5,150,105,0.2); border-radius: var(--radius-sm); padding: 10px 14px; color: #34d399; font-size: 0.82rem; margin: 8px 0; }
.t-err { background: rgba(220,38,38,0.1); border: 1px solid rgba(220,38,38,0.2); border-radius: var(--radius-sm); padding: 10px 14px; color: #f87171; font-size: 0.82rem; margin: 8px 0; }
.auth-card [data-testid="stTextInput"] > div > div > input { border-radius: var(--radius-sm) !important; background: rgba(255,255,255,0.04) !important; border: 1.5px solid rgba(255,255,255,0.08) !important; padding: 12px 16px !important; font-size: 0.88rem !important; color: var(--t1) !important; }
.auth-card [data-testid="stTextInput"] > div > div > input:focus { border-color: var(--red) !important; box-shadow: 0 0 0 3px rgba(229,9,20,0.1) !important; }
.auth-card [data-testid="stTextInput"] label { color: var(--t2) !important; font-size: 0.8rem !important; }
.auth-card .stButton > button { background: var(--red-grad) !important; color: #fff !important; border: none !important; border-radius: var(--radius-sm) !important; font-weight: 600 !important; width: 100% !important; }
.auth-secondary .stButton > button { background: rgba(255,255,255,0.04) !important; border: 1px solid var(--border-hi) !important; color: var(--t2) !important; box-shadow: none !important; }
</style>
"""

MAIN_CSS = """
<style>
.block-container, [data-testid="stMainBlockContainer"] { padding-top: 0 !important; padding-bottom: 2rem !important; max-width: 1400px !important; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, var(--bg2) 0%, #030711 100%) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] * { color: var(--t2) !important; }
[data-testid="stSidebar"] .stButton > button { background: rgba(255,255,255,0.03) !important; border: 1px solid var(--border) !important; color: var(--t1) !important; border-radius: var(--radius-sm) !important; font-weight: 500 !important; width: 100% !important; padding: 10px 16px !important; box-shadow: none !important; }
[data-testid="stSidebar"] .stButton > button:hover { background: rgba(229,9,20,0.08) !important; border-color: rgba(229,9,20,0.2) !important; color: #fff !important; }
.main > div > [data-testid="stTextInput"] > div > div > input { background: rgba(15,23,42,0.6) !important; border: 1.5px solid rgba(255,255,255,0.07) !important; border-radius: 50px !important; color: var(--t1) !important; font-size: 0.92rem !important; padding: 14px 24px !important; caret-color: var(--red) !important; }
.main > div > [data-testid="stTextInput"] > div > div > input:focus { border-color: var(--red) !important; box-shadow: 0 0 0 3px rgba(229,9,20,0.1) !important; }
[data-testid="stTextInput"] label { display: none !important; }
.main .stButton > button { background: var(--red-grad) !important; color: #fff !important; border: none !important; border-radius: var(--radius-sm) !important; font-weight: 600 !important; font-size: 0.76rem !important; padding: 7px 0 !important; width: 100% !important; }
.wl-a button { background: linear-gradient(135deg,#059669,#047857) !important; font-size: 0.7rem !important; padding: 5px 0 !important; }
.wl-r button { background: rgba(220,38,38,0.12) !important; border: 1px solid rgba(220,38,38,0.2) !important; box-shadow: none !important; font-size: 0.7rem !important; color: #f87171 !important; }
.wl-da button { background: linear-gradient(135deg,#059669,#047857) !important; font-size: 0.85rem !important; padding: 10px 0 !important; }
.wl-dr button { background: rgba(71,85,105,0.3) !important; border: 1px solid rgba(255,255,255,0.1) !important; font-size: 0.85rem !important; padding: 10px 0 !important; box-shadow: none !important; }
.um button { background: rgba(255,255,255,0.04) !important; border: 1px solid var(--border) !important; color: var(--t2) !important; border-radius: var(--radius-sm) !important; box-shadow: none !important; }
.um-lo button { background: rgba(220,38,38,0.06) !important; border: 1px solid rgba(220,38,38,0.15) !important; color: #f87171 !important; box-shadow: none !important; }
[data-testid="stHorizontalBlock"] { gap: 0.75rem !important; }
hr { border: none !important; border-top: 1px solid var(--border) !important; }
[data-testid="stSelectbox"] > div > div { background: var(--surface) !important; border: 1.5px solid var(--border) !important; border-radius: var(--radius-sm) !important; color: var(--t1) !important; }
[data-testid="stExpander"] { background: var(--surface-hi) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important; }
[data-testid="stExpander"] summary { color: var(--t2) !important; font-size: 0.82rem !important; }
[data-testid="stAlert"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important; color: var(--t2) !important; }
</style>
"""

st.markdown(BASE_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# AUTH PAGE (preserved from original)
# ═══════════════════════════════════════════════════════════

if not st.session_state.logged_in:
    st.markdown(AUTH_CSS, unsafe_allow_html=True)
    st.markdown("<div class='auth-page'>", unsafe_allow_html=True)
    spacer_l, form_col, spacer_r = st.columns([1.5, 1, 1.5])

    with form_col:
        st.markdown("<div class='auth-logo'><div class='auth-ico'>🎬</div></div>", unsafe_allow_html=True)

        if st.session_state.auth_page == "login":
            st.markdown("<div class='auth-h'>Welcome Back</div>", unsafe_allow_html=True)
            st.markdown("<div class='auth-sub'>Sign in to your personalized movie experience</div>", unsafe_allow_html=True)
            email_in = st.text_input("Email", placeholder="your@email.com", key="login_email")
            pass_in = st.text_input("Password", type="password", placeholder="Enter password", key="login_pass")
            if st.button("Sign In", use_container_width=True, key="btn_login"):
                if email_in and pass_in:
                    ok, result = login_user(email_in.strip(), pass_in)
                    if ok:
                        save_session(result["email"], result["name"])
                        st.session_state.logged_in = True
                        st.session_state.user_email = result["email"]
                        st.session_state.user_name = result["name"]
                        # Sync watchlist to collaborative engine
                        wl = get_watchlist(result["email"])
                        if wl:
                            titles = [m.get("title", "") for m in wl if m.get("title")]
                            api_post_json("/track/sync-watchlist", {"user_email": result["email"], "titles": titles})
                        st.rerun()
                    else:
                        st.markdown(f"<div class='t-err'>❌ {result}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='t-err'>❌ Please fill in all fields.</div>", unsafe_allow_html=True)
            st.markdown("<div class='auth-or'>or</div>", unsafe_allow_html=True)
            st.markdown("<div class='auth-secondary'>", unsafe_allow_html=True)
            if st.button("Create a new account", use_container_width=True, key="goto_signup"):
                st.session_state.auth_page = "signup"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='auth-h'>Create Account</div>", unsafe_allow_html=True)
            st.markdown("<div class='auth-sub'>Join Movie AI and discover amazing films</div>", unsafe_allow_html=True)
            name_in = st.text_input("Full Name", placeholder="John Doe", key="signup_name")
            email_in = st.text_input("Email", placeholder="your@email.com", key="signup_email")
            pass_in = st.text_input("Password", type="password", placeholder="Create password", key="signup_pass")
            pass2_in = st.text_input("Confirm Password", type="password", placeholder="Confirm password", key="signup_pass2")
            if st.button("Create Account", use_container_width=True, key="btn_signup"):
                if name_in and email_in and pass_in and pass2_in:
                    if pass_in != pass2_in:
                        st.markdown("<div class='t-err'>❌ Passwords don't match.</div>", unsafe_allow_html=True)
                    elif len(pass_in) < 4:
                        st.markdown("<div class='t-err'>❌ Password must be at least 4 characters.</div>", unsafe_allow_html=True)
                    else:
                        ok, msg = signup_user(name_in.strip(), email_in.strip(), pass_in)
                        if ok:
                            st.markdown(f"<div class='t-ok'>✅ {msg} Please sign in.</div>", unsafe_allow_html=True)
                            st.session_state.auth_page = "login"
                            st.rerun()
                        else:
                            st.markdown(f"<div class='t-err'>❌ {msg}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='t-err'>❌ Please fill in all fields.</div>", unsafe_allow_html=True)
            st.markdown("<div class='auth-or'>or</div>", unsafe_allow_html=True)
            st.markdown("<div class='auth-secondary'>", unsafe_allow_html=True)
            if st.button("Back to Sign In", use_container_width=True, key="goto_login"):
                st.session_state.auth_page = "login"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# ═══════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════

st.markdown(MAIN_CSS, unsafe_allow_html=True)

user_name = st.session_state.user_name or "User"
user_initial = user_name[0].upper()

# Navbar
st.markdown(f"""
<div class="nav"><div class="nav-in">
    <div class="brand"><div class="brand-i">🎬</div><span class="brand-t">Movie AI</span></div>
    <div class="nav-rt">
        <span class="nav-nm">{user_name}</span>
        <div class="nav-av">{user_initial}</div>
    </div>
</div></div>
""", unsafe_allow_html=True)

# User dropdown
_, _, menu_col = st.columns([5, 1.5, 1.2])
with menu_col:
    with st.expander(f"👤 {user_name}", expanded=False):
        st.markdown("<div class='um'>", unsafe_allow_html=True)
        if st.button("📋 Watchlist", key="um_wl", use_container_width=True):
            goto_watchlist()
        if st.button("📊 Model Insights", key="um_insights", use_container_width=True):
            goto_insights()
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='um-lo'>", unsafe_allow_html=True)
        if st.button("🚪 Logout", key="um_logout", use_container_width=True):
            do_logout()
        st.markdown("</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown(f"""
    <div style="padding:4px 0 12px">
        <div style="display:flex;align-items:center;gap:10px">
            <div style="background:var(--red-grad);width:32px;height:32px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:1rem;box-shadow:0 0 16px var(--red-glow)">🎬</div>
            <span style="font-size:1.05rem;font-weight:800;color:#f1f5f9">Movie AI</span>
        </div>
        <div style="font-size:0.78rem;color:#64748b;margin-top:8px">Welcome, <span style="color:#e2e8f0;font-weight:600">{user_name}</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    if st.button("🏠  Home", key="sb_home", use_container_width=True):
        goto_home()
    wl_count = len(get_watchlist(st.session_state.user_email))
    if st.button(f"❤️  Watchlist ({wl_count})", key="sb_wl", use_container_width=True):
        goto_watchlist()
    if st.button("📊  Model Insights", key="sb_insights", use_container_width=True):
        goto_insights()

    st.markdown("<div style='font-size:0.7rem;color:#475569;letter-spacing:1px;text-transform:uppercase;font-weight:600;margin:20px 0 8px'>🎞️ Category</div>", unsafe_allow_html=True)
    cat_keys = list(CATS.keys())
    cat_labels = [f"{CATS[k][1]}  {CATS[k][0]}" for k in cat_keys]
    current_idx = cat_keys.index(st.session_state.category) if st.session_state.category in cat_keys else 0
    selected_label = st.selectbox("Category", cat_labels, index=current_idx, label_visibility="collapsed", key="cat_select")
    new_cat = cat_keys[cat_labels.index(selected_label)] if selected_label in cat_labels else "trending"
    if new_cat != st.session_state.category:
        st.session_state.category = new_cat
        st.rerun()

    st.markdown("<div style='font-size:0.7rem;color:#475569;letter-spacing:1px;text-transform:uppercase;font-weight:600;margin:20px 0 8px'>⚙️ Grid</div>", unsafe_allow_html=True)
    grid_cols = st.slider("Grid", 4, 8, 6, label_visibility="collapsed")
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    if st.button("🚪 Logout", key="sb_logout", use_container_width=True):
        do_logout()

# Search Bar
_, search_col, _ = st.columns([0.6, 3, 0.6])
with search_col:
    typed = st.text_input("search", placeholder="Search movies... e.g. Avengers, Inception, Batman", label_visibility="collapsed")
    # Track searches
    if typed.strip() and st.session_state.user_email:
        _track_interaction(st.session_state.user_email, typed.strip(), "search")

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════

if st.session_state.view == "home":
    if typed.strip():
        query = typed.strip()
        if len(query) < 2:
            st.caption("Type at least 2 characters.")
        else:
            rs = st.session_state.recent_searches
            if query not in rs:
                rs.insert(0, query)
                st.session_state.recent_searches = rs[:8]
            with st.spinner(""):
                data, err = api_get_json("/tmdb/search", params={"query": query})
            if err or data is None:
                st.error(f"Search failed: {err}")
            else:
                suggestions, cards = parse_search(data, query, 24)
                if suggestions:
                    labels = ["-- Select a movie --"] + [s[0] for s in suggestions]
                    pick = st.selectbox("💡 Quick pick", labels, index=0)
                    if pick != "-- Select a movie --":
                        mapping = {s[0]: s[1] for s in suggestions}
                        goto_details(mapping[pick])
                st.markdown(f'<div class="sh fi"><span class="e">🔍</span><span class="l">Results for "{query}"</span><span class="b">{len(cards)} found</span></div>', unsafe_allow_html=True)
                render_poster_grid(cards, cols=grid_cols, kp="search", show_wl=True)
        st.stop()

    # ── NEW: "Recommended for You" section (hybrid) ──
    email = st.session_state.user_email
    if email:
        with st.spinner(""):
            fy_data, fy_err = api_get_json("/recommend/for-you", params={"user_email": email, "top_n": 12})
        if not fy_err and fy_data and len(fy_data) > 0:
            fy_cards = to_hybrid_cards(fy_data)
            if fy_cards:
                st.markdown('<div class="sh fi"><span class="e">🤖</span><span class="l">Recommended for You</span><span class="b">Hybrid AI</span></div>', unsafe_allow_html=True)
                st.markdown("<p style='color:var(--t3);font-size:0.82rem;margin:-8px 0 12px'>Personalized picks powered by content analysis + collaborative filtering</p>", unsafe_allow_html=True)
                render_poster_grid(fy_cards, cols=grid_cols, kp="foryou", show_wl=True, show_reason=True)

    # Recent searches
    if st.session_state.recent_searches:
        st.markdown('<div class="sh fi"><span class="e">🕐</span><span class="l">Recent Searches</span></div>', unsafe_allow_html=True)
        chip_cols = st.columns(min(len(st.session_state.recent_searches), 8))
        for i, q in enumerate(st.session_state.recent_searches[:8]):
            with chip_cols[i % len(chip_cols)]:
                st.markdown(f"<div class='chip' style='text-align:center;width:100%'>{q}</div>", unsafe_allow_html=True)

    # Category feed
    cat_key = st.session_state.category
    cat_label, cat_icon = CATS.get(cat_key, ("Trending", "🔥"))
    st.markdown(f'<div class="sh fi"><span class="e">{cat_icon}</span><span class="l">{cat_label}</span><span class="b">Home Feed</span></div>', unsafe_allow_html=True)
    with st.spinner(""):
        home_cards, err = api_get_json("/home", params={"category": cat_key, "limit": 24})
    if err or not home_cards:
        st.error(f"Home feed failed: {err or 'Unknown error'}")
        st.stop()
    render_poster_grid(home_cards, cols=grid_cols, kp="home", show_wl=True)


# ═══════════════════════════════════════════════════════════
# PAGE: WATCHLIST
# ═══════════════════════════════════════════════════════════

elif st.session_state.view == "watchlist":
    watchlist = get_watchlist(st.session_state.user_email)
    st.markdown(f'<div class="sh fi"><span class="e">❤️</span><span class="l">My Watchlist</span><span class="b">{len(watchlist)} movies</span></div>', unsafe_allow_html=True)

    if not watchlist:
        st.markdown('<div class="empty fi"><div class="empty-i">🎬</div><div class="empty-h">Your watchlist is empty</div><div class="empty-p">Browse movies and add them to your watchlist.</div></div>', unsafe_allow_html=True)
        if st.button("🏠 Browse Movies", key="wl_browse"):
            goto_home()
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="stat fi"><div class="stat-n">{len(watchlist)}</div><div class="stat-l">Total Movies</div></div>', unsafe_allow_html=True)
        with c2:
            posters = sum(1 for m in watchlist if m.get("poster_url"))
            st.markdown(f'<div class="stat fi fi1"><div class="stat-n">{posters}</div><div class="stat-l">With Posters</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat fi fi2"><div class="stat-n">~{len(watchlist) * 2}h</div><div class="stat-l">Watch Time</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        render_poster_grid(watchlist, cols=grid_cols, kp="wl", show_wl=True)

        # ── NEW: "Based on Your Watchlist" (collaborative) ──
        email = st.session_state.user_email
        if email and len(watchlist) >= 2:
            with st.spinner(""):
                collab_data, collab_err = api_get_json("/recommend/collaborative", params={"user_email": email, "top_n": 12})
            if not collab_err and collab_data:
                collab_cards = to_hybrid_cards(collab_data) if isinstance(collab_data, list) else []
                if not collab_cards:
                    # Fallback: parse as flat list
                    collab_cards = []
                    for x in collab_data:
                        t = x.get("tmdb") or {}
                        if t.get("tmdb_id"):
                            collab_cards.append({"tmdb_id": t["tmdb_id"], "title": t.get("title", x.get("title", "")), "poster_url": t.get("poster_url")})
                if collab_cards:
                    st.markdown('<div class="sh fi"><span class="e">👥</span><span class="l">Based on Your Watchlist</span><span class="b">Collaborative</span></div>', unsafe_allow_html=True)
                    st.markdown("<p style='color:var(--t3);font-size:0.82rem;margin:-8px 0 12px'>Movies enjoyed by users with similar taste</p>", unsafe_allow_html=True)
                    render_poster_grid(collab_cards, cols=grid_cols, kp="collab_wl", show_wl=True)


# ═══════════════════════════════════════════════════════════
# PAGE: DETAILS
# ═══════════════════════════════════════════════════════════

elif st.session_state.view == "details":
    tmdb_id = st.session_state.selected_tmdb_id
    if not tmdb_id:
        st.warning("No movie selected.")
        if st.button("← Home"):
            goto_home()
        st.stop()

    back_col, _ = st.columns([1.2, 5])
    with back_col:
        if st.button("← Back to Home"):
            goto_home()

    with st.spinner(""):
        data, err = api_get_json(f"/movie/id/{tmdb_id}")

    if err or not data:
        st.error(f"Could not load movie: {err or 'Unknown error'}")
        st.stop()

    # Track this click
    email = st.session_state.user_email
    if email:
        genres = [g.get("name", "") for g in data.get("genres", [])]
        _track_interaction(email, data.get("title", ""), "click", tmdb_id=tmdb_id, genres=genres)

    # Backdrop
    if data.get("backdrop_url"):
        st.markdown(f'<div class="dh fi"><img src="{data["backdrop_url"]}"/><div class="dh-ov"></div><div class="dh-s"></div></div>', unsafe_allow_html=True)

    left_col, right_col = st.columns([1.2, 3], gap="large")

    with left_col:
        st.markdown("<div class='pw fi fi1'>", unsafe_allow_html=True)
        if data.get("poster_url"):
            safe_image(data["poster_url"])
        else:
            st.markdown("<div class='no-p' style='height:400px'><div class='no-p-i'>🎬</div><span>No Poster</span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if email:
            if is_in_watchlist(email, tmdb_id):
                st.markdown("<div class='wl-dr'>", unsafe_allow_html=True)
                if st.button("✓ In Watchlist (Remove)", key="detail_wl_remove"):
                    remove_from_watchlist(email, int(tmdb_id))
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='wl-da'>", unsafe_allow_html=True)
                if st.button("♡ Add to Watchlist", key="detail_wl_add"):
                    add_to_watchlist(email, {"tmdb_id": int(tmdb_id), "title": data.get("title", ""), "poster_url": data.get("poster_url")})
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        title = data.get("title", "")
        release = data.get("release_date") or "—"
        genres = data.get("genres", [])
        overview = data.get("overview") or "No overview available."

        chips_html = ""
        if release != "—":
            yr = release[:4] if len(release) >= 4 else release
            chips_html += f'<span class="chip chip-a">📅 {yr}</span>'
        for g in genres:
            chips_html += f'<span class="chip">{g["name"]}</span>'

        st.markdown(f"""
        <div class="dc fi fi2">
            <div class="dt">{title}</div>
            <div class="dm">{chips_html}</div>
            <div class="ovl">Overview</div>
            <div class="ovt">{overview}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── NEW: Hybrid Recommendations (with explanations) ──
    movie_title = (data.get("title") or "").strip()
    if movie_title and email:
        st.markdown('<div class="sh fi fi3"><span class="e">🤖</span><span class="l">AI-Powered Recommendations</span><span class="b">Hybrid</span></div>', unsafe_allow_html=True)
        st.markdown(f"<p style='color:var(--t3);font-size:0.82rem;margin:-8px 0 12px'>Content similarity + collaborative patterns for \"{movie_title}\"</p>", unsafe_allow_html=True)

        with st.spinner(""):
            hybrid_data, hybrid_err = api_get_json("/recommend/hybrid", params={"user_email": email, "movie_title": movie_title, "top_n": 12})

        if not hybrid_err and hybrid_data:
            hybrid_cards = to_hybrid_cards(hybrid_data)
            if hybrid_cards:
                render_poster_grid(hybrid_cards, cols=grid_cols, kp="hybrid_det", show_wl=True, show_reason=True)

    # TF-IDF Similar Movies (preserved)
    if movie_title:
        with st.spinner(""):
            bundle, err2 = api_get_json("/movie/search", params={"query": movie_title, "tfidf_top_n": 12, "genre_limit": 12})

        if not err2 and bundle:
            tfidf_cards = to_tfidf_cards(bundle.get("tfidf_recommendations"))
            if tfidf_cards:
                st.markdown('<div class="sh fi"><span class="e">🔎</span><span class="l">Similar Movies</span><span class="b">TF-IDF Content</span></div>', unsafe_allow_html=True)
                st.markdown("<p style='color:var(--t3);font-size:0.82rem;margin:-8px 0 12px'>Based on content similarity analysis (genres, keywords, cast)</p>", unsafe_allow_html=True)
                render_poster_grid(tfidf_cards, cols=grid_cols, kp="rec_tf", show_wl=True)

            genre_cards = bundle.get("genre_recommendations", [])
            if genre_cards:
                st.markdown('<div class="sh fi"><span class="e">🎭</span><span class="l">More Like This</span><span class="b">Genre</span></div>', unsafe_allow_html=True)
                render_poster_grid(genre_cards, cols=grid_cols, kp="rec_gn", show_wl=True)


# ═══════════════════════════════════════════════════════════
# PAGE: MODEL INSIGHTS (NEW)
# ═══════════════════════════════════════════════════════════

elif st.session_state.view == "insights":
    st.markdown('<div class="sh fi"><span class="e">📊</span><span class="l">Model Insights</span><span class="b">Analytics</span></div>', unsafe_allow_html=True)
    st.markdown("<p style='color:var(--t3);font-size:0.88rem;margin:-8px 0 20px'>Evaluation metrics, algorithm explanations, and system performance</p>", unsafe_allow_html=True)

    # ── Metrics ──
    with st.spinner("Loading metrics..."):
        metrics, m_err = api_get_json("/metrics", params={"k": 10})

    if m_err or not metrics:
        st.warning(f"Could not load metrics: {m_err or 'Unknown'}")
    else:
        st.markdown('<div class="sh fi"><span class="e">🎯</span><span class="l">Evaluation Metrics</span><span class="b">@K=10</span></div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            val = metrics.get("avg_precision", 0)
            st.markdown(f'<div class="metric-card"><div class="metric-title">Precision@10</div><div class="metric-val">{val:.2%}</div><div class="metric-sub">Fraction of relevant recs</div></div>', unsafe_allow_html=True)
        with c2:
            val = metrics.get("avg_recall", 0)
            st.markdown(f'<div class="metric-card"><div class="metric-title">Recall@10</div><div class="metric-val">{val:.2%}</div><div class="metric-sub">Relevant items found</div></div>', unsafe_allow_html=True)
        with c3:
            val = metrics.get("avg_ndcg", 0)
            st.markdown(f'<div class="metric-card"><div class="metric-title">NDCG@10</div><div class="metric-val">{val:.2%}</div><div class="metric-sub">Ranking quality</div></div>', unsafe_allow_html=True)
        with c4:
            val = metrics.get("hit_rate", 0)
            st.markdown(f'<div class="metric-card"><div class="metric-title">Hit Rate</div><div class="metric-val">{val:.2%}</div><div class="metric-sub">Held-out item found</div></div>', unsafe_allow_html=True)

        rmse_val = metrics.get("avg_rmse")
        cov = metrics.get("coverage_pct", 0)
        c5, c6, c7 = st.columns(3)
        with c5:
            rmse_display = f"{rmse_val:.4f}" if rmse_val is not None else "N/A"
            st.markdown(f'<div class="metric-card"><div class="metric-title">RMSE</div><div class="metric-val">{rmse_display}</div><div class="metric-sub">Prediction error</div></div>', unsafe_allow_html=True)
        with c6:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Catalog Coverage</div><div class="metric-val">{cov:.1f}%</div><div class="metric-sub">Unique movies recommended</div></div>', unsafe_allow_html=True)
        with c7:
            n_users = metrics.get("num_users_evaluated", 0)
            st.markdown(f'<div class="metric-card"><div class="metric-title">Users Evaluated</div><div class="metric-val">{n_users}</div><div class="metric-sub">With sufficient data</div></div>', unsafe_allow_html=True)

        # Collaborative stats
        collab_stats = metrics.get("collab_stats", {})
        if collab_stats:
            st.markdown('<div class="sh fi"><span class="e">👥</span><span class="l">Collaborative Engine Stats</span></div>', unsafe_allow_html=True)
            cc1, cc2, cc3, cc4 = st.columns(4)
            with cc1:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Users</div><div class="metric-val">{collab_stats.get("total_users", 0)}</div></div>', unsafe_allow_html=True)
            with cc2:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Interactions</div><div class="metric-val">{collab_stats.get("total_interactions", 0)}</div></div>', unsafe_allow_html=True)
            with cc3:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Matrix</div><div class="metric-val" style="font-size:1.2rem">{collab_stats.get("matrix_shape", "N/A")}</div></div>', unsafe_allow_html=True)
            with cc4:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Sparsity</div><div class="metric-val" style="font-size:1.2rem">{collab_stats.get("sparsity", "N/A")}</div></div>', unsafe_allow_html=True)

    # ── User Profile Stats ──
    email = st.session_state.user_email
    if email:
        user_stats, us_err = api_get_json("/user/stats", params={"user_email": email})
        if not us_err and user_stats:
            st.markdown('<div class="sh fi"><span class="e">👤</span><span class="l">Your Activity</span></div>', unsafe_allow_html=True)
            uc1, uc2, uc3, uc4 = st.columns(4)
            with uc1:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Clicks</div><div class="metric-val">{user_stats.get("total_clicks", 0)}</div></div>', unsafe_allow_html=True)
            with uc2:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Watchlist</div><div class="metric-val">{user_stats.get("total_watchlist", 0)}</div></div>', unsafe_allow_html=True)
            with uc3:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Searches</div><div class="metric-val">{user_stats.get("total_searches", 0)}</div></div>', unsafe_allow_html=True)
            with uc4:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Genres Explored</div><div class="metric-val">{user_stats.get("genres_explored", 0)}</div></div>', unsafe_allow_html=True)

        # Top genres
        top_genres, tg_err = api_get_json("/user/top-genres", params={"user_email": email, "top_n": 5})
        if not tg_err and top_genres:
            st.markdown('<div class="sh fi"><span class="e">🎭</span><span class="l">Your Top Genres</span></div>', unsafe_allow_html=True)
            genre_cols = st.columns(min(len(top_genres), 5))
            for i, g in enumerate(top_genres):
                with genre_cols[i]:
                    st.markdown(f'<div class="metric-card" style="text-align:center"><div class="metric-val" style="font-size:1.4rem">{g["count"]}</div><div class="metric-sub">{g["genre"]}</div></div>', unsafe_allow_html=True)

    # ── Algorithm Explanations ──
    algos, algo_err = api_get_json("/metrics/algorithms")
    if not algo_err and algos:
        st.markdown('<div class="sh fi"><span class="e">🧠</span><span class="l">Algorithm Explanations</span></div>', unsafe_allow_html=True)

        names = {
            "tfidf_content": "TF-IDF Content-Based Filtering",
            "collaborative_svd": "Collaborative Filtering (SVD)",
            "hybrid": "Hybrid Recommendation Engine",
            "explainability": "Explainable AI (XAI)",
        }
        for key, desc in algos.items():
            name = names.get(key, key)
            st.markdown(f"""
            <div class="algo-card">
                <div class="algo-name">{name}</div>
                <div class="algo-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
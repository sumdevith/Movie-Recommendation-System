"""
FastAPI demo for the Demographic + Genre movie recommender.

The user supplies a demographic profile (Gender, Age, Occupation, Zip-code) and
1-3 preferred genres plus how many recommendations they want. The server scores
every movie that belongs to the chosen genres with the trained model and returns
the top-N highest-predicted titles.
"""

from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import features as feat
from section2_model_architecture import build_model


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_for_streamlit.pt"
MOVIES_PATH = BASE_DIR / "data" / "movies.dat"


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    gender: str = Field(..., description="'M' or 'F'")
    age: int = Field(..., description="MovieLens age code: 1,18,25,35,45,50,56")
    occupation: int = Field(..., ge=0, le=20)
    zip_code: str = Field(..., description="US zip-code (only the leading digit is used)")
    genres: list[str] = Field(..., min_length=1, max_length=3)
    top_n: int = Field(10, ge=1, le=50)


class MovieRecommendation(BaseModel):
    movie_id: int
    title: str
    genres: str
    predicted_rating: float


class RecommendationResponse(BaseModel):
    top_n: int
    profile: dict
    recommendations: list[MovieRecommendation]


app = FastAPI(title="MovieLens Demographic+Genre Recommender")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


# ---------------------------------------------------------------------------
# Asset loading (movies catalogue + trained model)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_movies() -> pd.DataFrame:
    if not MOVIES_PATH.exists():
        raise RuntimeError(f"Required file not found: {MOVIES_PATH}")
    movies = pd.read_csv(
        MOVIES_PATH, sep="::", engine="python",
        names=["movie_id", "title", "genres"], encoding="latin-1",
    )
    # Pre-compute each movie's genre multi-hot once.
    movies["genre_set"] = movies["genres"].apply(
        lambda s: {g.strip() for g in str(s).split("|")}
    )
    movies["multihot"] = movies["genres"].apply(feat.genres_to_multihot)
    return movies


@lru_cache(maxsize=1)
def load_model():
    """Load the trained checkpoint. Returns None if it has not been trained yet,
    or if the checkpoint predates the demographic+genre architecture."""
    if not MODEL_PATH.exists():
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODEL_PATH, map_location=device)

    # Older (user/movie-ID) checkpoints lack feature_dims and are incompatible.
    if not isinstance(ckpt, dict) or "feature_dims" not in ckpt:
        print("[app] Existing checkpoint is incompatible with the demographic+genre "
              "model — retrain and overwrite model_for_streamlit.pt.")
        return None

    model = build_model(
        model_type   = ckpt["model_type"],
        feature_dims = ckpt["feature_dims"],
        embed_dim    = ckpt["embed_dim"],
        hidden_dims  = ckpt["hidden_dims"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return {"model": model, "device": device}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.get("/api/options")
def options():
    """Vocabulary for the UI dropdowns — kept in sync with features.py."""
    return {
        "genders": [{"value": "M", "label": "Male"}, {"value": "F", "label": "Female"}],
        "ages": [{"value": code, "label": feat.AGE_LABELS[idx]} for code, idx in feat.AGE_MAP.items()],
        "occupations": [{"value": k, "label": v} for k, v in feat.OCCUPATION_LABELS.items()],
        "genres": feat.GENRES,
    }


@app.get("/api/health")
def health():
    model_loaded = load_model() is not None
    return {
        "ok": True,
        "model": MODEL_PATH.name,
        "model_loaded": model_loaded,
        "movies": int(len(load_movies())),
    }


@app.post("/api/recommend", response_model=RecommendationResponse)
def recommend(req: RecommendRequest):
    assets = load_model()
    if assets is None:
        raise HTTPException(
            status_code=503,
            detail=("Model not trained yet. Train it (Colab) and place the checkpoint "
                    f"at '{MODEL_PATH.name}', then retry."),
        )

    if req.gender not in feat.GENDER_MAP:
        raise HTTPException(status_code=422, detail="gender must be 'M' or 'F'.")
    if req.age not in feat.AGE_MAP:
        raise HTTPException(status_code=422, detail=f"age must be one of {list(feat.AGE_MAP)}.")
    unknown = [g for g in req.genres if g not in feat.GENRE_TO_IDX]
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown genre(s): {unknown}.")

    movies = load_movies()
    wanted = set(req.genres)

    # Candidate movies = those that contain at least one of the chosen genres.
    candidates = movies[movies["genre_set"].apply(lambda s: bool(s & wanted))].copy()
    if candidates.empty:
        return {"top_n": req.top_n, "profile": _profile(req), "recommendations": []}

    model = assets["model"]
    device = assets["device"]

    # Encode the (shared) demographic profile once, broadcast across candidates.
    n = len(candidates)
    gender = torch.full((n,), feat.GENDER_MAP[req.gender], dtype=torch.long, device=device)
    age    = torch.full((n,), feat.AGE_MAP[req.age],       dtype=torch.long, device=device)
    occ    = torch.full((n,), req.occupation,              dtype=torch.long, device=device)
    zipr   = torch.full((n,), feat.zip_to_region(req.zip_code), dtype=torch.long, device=device)
    genres = torch.tensor(candidates["multihot"].tolist(), dtype=torch.float32, device=device)

    with torch.no_grad():
        scores = model(gender, age, occ, zipr, genres).cpu().tolist()

    candidates["predicted_rating"] = scores
    top = candidates.sort_values("predicted_rating", ascending=False).head(req.top_n)

    recommendations = [
        MovieRecommendation(
            movie_id=int(row.movie_id),
            title=str(row.title),
            genres=str(row.genres),
            predicted_rating=round(float(row.predicted_rating), 3),
        )
        for row in top.itertuples(index=False)
    ]

    return {"top_n": req.top_n, "profile": _profile(req), "recommendations": recommendations}


def _profile(req: RecommendRequest) -> dict:
    return {
        "gender": feat.GENDER_LABELS[feat.GENDER_MAP[req.gender]],
        "age": feat.AGE_LABELS[feat.AGE_MAP[req.age]],
        "occupation": feat.OCCUPATION_LABELS[req.occupation],
        "zip_code": req.zip_code,
        "genres": req.genres,
    }

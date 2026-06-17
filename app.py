from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from section2_model_architecture import build_model


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_for_streamlit.pt"
RATINGS_PATH = BASE_DIR / "data" / "ratings.dat"
MOVIES_PATH = BASE_DIR / "data" / "movies.dat"


class MovieRecommendation(BaseModel):
    movie_id: int
    title: str
    genres: str
    predicted_rating: float


class RecommendationResponse(BaseModel):
    user_id: int
    top_k: int
    recommendations: list[MovieRecommendation]


app = FastAPI(title="MovieLens NCF Recommender")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


def _require_file(path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"Required file not found: {path}")


def normalize_state_dict(state_dict: dict) -> dict:
    keys = list(state_dict.keys())
    for prefix in ("module.", "model."):
        if keys and all(key.startswith(prefix) for key in keys):
            return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def infer_model_config(state_dict: dict, checkpoint=None) -> dict:
    checkpoint = checkpoint or {}
    keys = set(state_dict.keys())

    if "user_embedding.weight" in keys:
        model_type = checkpoint.get("model_type", "ncf")
        user_weight = state_dict["user_embedding.weight"]
        movie_weight = state_dict["movie_embedding.weight"]
        hidden_dims = [
            tensor.shape[0]
            for name, tensor in state_dict.items()
            if name.startswith("mlp.")
            and name.endswith(".weight")
            and tensor.ndim == 2
        ]
        return {
            "model_type": model_type,
            "num_users": checkpoint.get("num_users", user_weight.shape[0]),
            "num_movies": checkpoint.get("num_movies", movie_weight.shape[0]),
            "kwargs": {
                "embed_dim": checkpoint.get("embed_dim", user_weight.shape[1]),
                "hidden_dims": checkpoint.get("hidden_dims", hidden_dims),
                "dropout": checkpoint.get("dropout", 0.3),
            },
        }

    if "gmf_user_embedding.weight" in keys:
        return {
            "model_type": checkpoint.get("model_type", "neumf"),
            "num_users": checkpoint.get("num_users", state_dict["gmf_user_embedding.weight"].shape[0]),
            "num_movies": checkpoint.get("num_movies", state_dict["gmf_movie_embedding.weight"].shape[0]),
            "kwargs": {"dropout": checkpoint.get("dropout", 0.3)},
        }

    raise RuntimeError("Could not infer model architecture from checkpoint.")


@lru_cache(maxsize=1)
def load_assets():
    _require_file(MODEL_PATH)
    _require_file(RATINGS_PATH)
    _require_file(MOVIES_PATH)

    ratings = pd.read_csv(
        RATINGS_PATH,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    movies = pd.read_csv(
        MOVIES_PATH,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )

    user_ids = sorted(ratings["user_id"].unique().tolist())
    movie_ids = sorted(ratings["movie_id"].unique().tolist())
    user_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
    movie_map = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

    rated_movies_by_user = (
        ratings.groupby("user_id")["movie_id"]
        .apply(lambda values: set(values.tolist()))
        .to_dict()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint.to(device)
    elif isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("state_dict")
            or checkpoint.get("model_state_dict")
            or checkpoint.get("model")
        )
        if state_dict is None and any(key.endswith(".weight") for key in checkpoint.keys()):
            state_dict = checkpoint
        if state_dict is None:
            raise RuntimeError("Checkpoint does not contain a state_dict.")
        state_dict = normalize_state_dict(state_dict)

        config = infer_model_config(
            state_dict,
            checkpoint if state_dict is not checkpoint else None,
        )

        model = build_model(
            model_type=config["model_type"],
            num_users=config["num_users"],
            num_movies=config["num_movies"],
            **config["kwargs"],
        ).to(device)
        model.load_state_dict(state_dict)
    else:
        raise RuntimeError("Unsupported checkpoint format.")

    model.eval()

    return {
        "ratings": ratings,
        "movies": movies,
        "user_map": user_map,
        "movie_map": movie_map,
        "rated_movies_by_user": rated_movies_by_user,
        "model": model,
        "device": device,
    }


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.get("/api/health")
def health():
    assets = load_assets()
    return {
        "ok": True,
        "model": MODEL_PATH.name,
        "users": len(assets["user_map"]),
        "movies": len(assets["movie_map"]),
        "device": str(assets["device"]),
    }


@app.get("/api/recommendations", response_model=RecommendationResponse)
def recommendations(
    user_id: int = Query(..., ge=1),
    top_k: int = Query(10, ge=1, le=50),
):
    assets = load_assets()
    user_map = assets["user_map"]
    movie_map = assets["movie_map"]

    if user_id not in user_map:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown user_id {user_id}. Try a MovieLens user ID from 1 to 6040.",
        )

    movies = assets["movies"]
    rated_movies = assets["rated_movies_by_user"].get(user_id, set())
    candidate_movies = movies[
        movies["movie_id"].isin(movie_map.keys())
        & ~movies["movie_id"].isin(rated_movies)
    ].copy()

    if candidate_movies.empty:
        return {"user_id": user_id, "top_k": top_k, "recommendations": []}

    model = assets["model"]
    device = assets["device"]
    user_idx = user_map[user_id]
    movie_indices = [movie_map[movie_id] for movie_id in candidate_movies["movie_id"]]

    predictions = []
    batch_size = 4096
    with torch.no_grad():
        for start in range(0, len(movie_indices), batch_size):
            batch_movie_idx = movie_indices[start : start + batch_size]
            users_tensor = torch.full(
                (len(batch_movie_idx),), user_idx, dtype=torch.long, device=device
            )
            movies_tensor = torch.tensor(batch_movie_idx, dtype=torch.long, device=device)
            batch_scores = model(users_tensor, movies_tensor).detach().cpu()
            predictions.extend(batch_scores.tolist())

    candidate_movies["predicted_rating"] = predictions
    top_movies = candidate_movies.sort_values("predicted_rating", ascending=False).head(top_k)

    results = [
        MovieRecommendation(
            movie_id=int(row.movie_id),
            title=str(row.title),
            genres=str(row.genres),
            predicted_rating=round(float(row.predicted_rating), 3),
        )
        for row in top_movies.itertuples(index=False)
    ]

    return {"user_id": user_id, "top_k": top_k, "recommendations": results}

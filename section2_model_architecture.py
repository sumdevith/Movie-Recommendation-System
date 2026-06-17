"""
=============================================================================
SECTION 2: Deep Learning Architecture — Demographic + Genre Recommender
=============================================================================
The model predicts a rating from *content features* instead of user/movie IDs,
so it can recommend movies to a brand-new user who only supplies a demographic
profile and a few preferred genres (the classic "cold-start" setting).

Inputs (per sample)
───────────────────
  USER side                         MOVIE side
  ─────────                         ──────────
  gender      (2 categories)        genres   (18-dim multi-hot)
  age group   (7 categories)
  occupation  (21 categories)
  zip region  (11 categories)

Architecture overview
─────────────────────
  gender ─► Emb ┐
  age    ─► Emb ├─ concat ─► Linear ─► user_vec  (embed_dim)
  occ    ─► Emb │
  zip    ─► Emb ┘

  genres ─► Linear(18 → embed_dim) ─────────► movie_vec (embed_dim)

  NCF   :  concat(user_vec, movie_vec) ─► MLP tower ─► rating
  NeuMF :  GMF(user_vec ⊙ movie_vec)  ⊕  MLP(concat) ─► rating

This keeps the NCF / NeuMF duality of the original project while swapping the
ID embeddings for demographic + genre feature towers.
=============================================================================
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 2A. Shared feature towers (user demographics + movie genres)
# ---------------------------------------------------------------------------

class UserTower(nn.Module):
    """Embed the four demographic categoricals and fuse into a single vector."""

    def __init__(self, num_gender, num_age, num_occupation, num_zip, embed_dim):
        super().__init__()
        # A modest per-field embedding size keeps the tower small; we then
        # project the concatenation down to the shared ``embed_dim``.
        field_dim = max(8, embed_dim // 2)

        self.gender_emb = nn.Embedding(num_gender,     field_dim)
        self.age_emb    = nn.Embedding(num_age,        field_dim)
        self.occ_emb    = nn.Embedding(num_occupation, field_dim)
        self.zip_emb    = nn.Embedding(num_zip,        field_dim)

        self.project = nn.Linear(field_dim * 4, embed_dim)

    def forward(self, gender, age, occupation, zip_region):
        x = torch.cat(
            [
                self.gender_emb(gender),
                self.age_emb(age),
                self.occ_emb(occupation),
                self.zip_emb(zip_region),
            ],
            dim=-1,
        )
        return self.project(x)                     # (batch, embed_dim)


class MovieTower(nn.Module):
    """Project the 18-dim multi-hot genre vector into the shared space."""

    def __init__(self, num_genres, embed_dim):
        super().__init__()
        self.project = nn.Linear(num_genres, embed_dim)

    def forward(self, genres):
        return self.project(genres)                # (batch, embed_dim)


# ---------------------------------------------------------------------------
# 2B. Demographic NCF — pure MLP variant
# ---------------------------------------------------------------------------

class DemographicNCF(nn.Module):
    """Feature-based Neural Collaborative Filtering (MLP variant).

    Parameters
    ----------
    num_gender / num_age / num_occupation / num_zip / num_genres
        Cardinalities of each categorical / multi-hot feature.
    embed_dim   : shared dimensionality of the user and movie towers.
    hidden_dims : MLP hidden layer sizes (default [128, 64]).
    dropout     : dropout probability after each hidden activation.
    rating_min / rating_max : output clamping range.
    """

    def __init__(
        self,
        num_gender:     int,
        num_age:        int,
        num_occupation: int,
        num_zip:        int,
        num_genres:     int,
        embed_dim:      int   = 32,
        hidden_dims:    list  = None,
        dropout:        float = 0.3,
        rating_min:     float = 1.0,
        rating_max:     float = 5.0,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.rating_min = rating_min
        self.rating_max = rating_max

        self.user_tower  = UserTower(num_gender, num_age, num_occupation, num_zip, embed_dim)
        self.movie_tower = MovieTower(num_genres, embed_dim)

        # MLP tower over concat(user_vec, movie_vec)
        layers = []
        in_features = embed_dim * 2
        for out_features in hidden_dims:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = out_features
        self.mlp = nn.Sequential(*layers)

        self.output_layer = nn.Linear(in_features, 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.05)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, gender, age, occupation, zip_region, genres):
        user_vec  = self.user_tower(gender, age, occupation, zip_region)
        movie_vec = self.movie_tower(genres)

        x = torch.cat([user_vec, movie_vec], dim=-1)
        x = self.mlp(x)
        raw = self.output_layer(x).squeeze(-1)
        return torch.clamp(raw, self.rating_min, self.rating_max)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return super().__repr__() + f"\n\n[DemographicNCF] Trainable parameters: {self.count_parameters():,}"


# ---------------------------------------------------------------------------
# 2C. Demographic NeuMF — GMF + MLP (extended architecture)
# ---------------------------------------------------------------------------

class DemographicNeuMF(nn.Module):
    """Feature-based NeuMF: a GMF stream (element-wise product of the user and
    movie towers) fused with an MLP stream (concatenation through dense layers).
    Each stream owns its own towers for maximum expressiveness.
    """

    def __init__(
        self,
        num_gender:      int,
        num_age:         int,
        num_occupation:  int,
        num_zip:         int,
        num_genres:      int,
        embed_dim:       int   = 32,
        mlp_hidden_dims: list  = None,
        dropout:         float = 0.3,
        rating_min:      float = 1.0,
        rating_max:      float = 5.0,
    ):
        super().__init__()

        if mlp_hidden_dims is None:
            mlp_hidden_dims = [128, 64]

        self.rating_min = rating_min
        self.rating_max = rating_max

        # ---- GMF towers ----
        self.gmf_user  = UserTower(num_gender, num_age, num_occupation, num_zip, embed_dim)
        self.gmf_movie = MovieTower(num_genres, embed_dim)

        # ---- MLP towers ----
        self.mlp_user  = UserTower(num_gender, num_age, num_occupation, num_zip, embed_dim)
        self.mlp_movie = MovieTower(num_genres, embed_dim)

        # ---- MLP tower ----
        mlp_layers = []
        in_dim = embed_dim * 2
        for out_dim in mlp_hidden_dims:
            mlp_layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = out_dim
        self.mlp = nn.Sequential(*mlp_layers)

        # ---- Prediction layer: GMF (embed_dim) + MLP final hidden ----
        self.predict = nn.Linear(embed_dim + in_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.05)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, gender, age, occupation, zip_region, genres):
        # GMF stream — element-wise product
        gmf_u = self.gmf_user(gender, age, occupation, zip_region)
        gmf_m = self.gmf_movie(genres)
        gmf_out = gmf_u * gmf_m                     # (batch, embed_dim)

        # MLP stream — concatenation through dense tower
        mlp_u = self.mlp_user(gender, age, occupation, zip_region)
        mlp_m = self.mlp_movie(genres)
        mlp_out = self.mlp(torch.cat([mlp_u, mlp_m], dim=-1))

        combined = torch.cat([gmf_out, mlp_out], dim=-1)
        raw = self.predict(combined).squeeze(-1)
        return torch.clamp(raw, self.rating_min, self.rating_max)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 2D. Model factory — pick NCF or NeuMF by name
# ---------------------------------------------------------------------------

def build_model(
    model_type:   str,
    feature_dims: dict,
    **kwargs,
) -> nn.Module:
    """Factory function.

    Parameters
    ----------
    model_type   : "ncf"  → DemographicNCF (pure MLP)
                   "neumf" → DemographicNeuMF (GMF + MLP)
    feature_dims : dict with keys num_gender, num_age, num_occupation,
                   num_zip, num_genres (see ``features.feature_dims``).
    **kwargs     : forwarded to the chosen model constructor
                   (embed_dim, dropout, and hidden_dims / mlp_hidden_dims).

    Returns
    -------
    nn.Module
    """
    model_type = model_type.lower()

    common = dict(
        num_gender     = feature_dims["num_gender"],
        num_age        = feature_dims["num_age"],
        num_occupation = feature_dims["num_occupation"],
        num_zip        = feature_dims["num_zip"],
        num_genres     = feature_dims["num_genres"],
    )

    if model_type == "ncf":
        model = DemographicNCF(**common, **kwargs)
    elif model_type == "neumf":
        # NeuMF uses ``mlp_hidden_dims`` rather than ``hidden_dims``.
        if "hidden_dims" in kwargs:
            kwargs["mlp_hidden_dims"] = kwargs.pop("hidden_dims")
        model = DemographicNeuMF(**common, **kwargs)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'ncf' or 'neumf'.")

    print(
        f"\n[Model] Built '{model_type.upper()}'  "
        f"| Features: gender={common['num_gender']}, age={common['num_age']}, "
        f"occ={common['num_occupation']}, zip={common['num_zip']}, "
        f"genres={common['num_genres']}  "
        f"| Params: {model.count_parameters():,}\n"
    )
    return model

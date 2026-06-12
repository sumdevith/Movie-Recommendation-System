"""
=============================================================================
SECTION 2: Deep Learning Architecture — Neural Collaborative Filtering (NCF)
=============================================================================
Architecture overview
─────────────────────
  User ID  ──► UserEmbedding(num_users,  embed_dim)  ──►  u_vec
  Movie ID ──► MovieEmbedding(num_movies, embed_dim)  ──►  m_vec

  concat([u_vec, m_vec])                              ──►  x  (2 * embed_dim)
    │
    ├─ Linear(2*embed_dim → hidden[0]) + BatchNorm + ReLU + Dropout
    ├─ Linear(hidden[0]   → hidden[1]) + BatchNorm + ReLU + Dropout
    ├─ Linear(hidden[1]   → hidden[2]) + BatchNorm + ReLU + Dropout  (optional)
    └─ Linear(hidden[-1]  → 1)                        ──►  rating prediction

The design follows the "NeuMF" style from He et al. (2017) — pure MLP branch.
=============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 2A. Neural Collaborative Filtering Model
# ---------------------------------------------------------------------------

class NCFModel(nn.Module):
    """
    Neural Collaborative Filtering (NCF) — MLP variant.

    Parameters
    ----------
    num_users   : total number of unique users  (size of user embedding table)
    num_movies  : total number of unique movies (size of movie embedding table)
    embed_dim   : dimensionality of each embedding vector (default 64)
    hidden_dims : list of hidden layer sizes for the MLP tower (default [256, 128, 64])
    dropout     : dropout probability applied after each hidden activation (default 0.3)
    rating_min  : minimum rating value — used to clamp the output (default 1.0)
    rating_max  : maximum rating value — used to clamp the output (default 5.0)
    """

    def __init__(
        self,
        num_users:   int,
        num_movies:  int,
        embed_dim:   int   = 64,
        hidden_dims: list  = None,
        dropout:     float = 0.3,
        rating_min:  float = 1.0,
        rating_max:  float = 5.0,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]   # 3 hidden layers by default

        self.embed_dim  = embed_dim
        self.rating_min = rating_min
        self.rating_max = rating_max

        # ------------------------------------------------------------------
        # Embedding Layers
        # ------------------------------------------------------------------
        # Each user ID is mapped to a dense embed_dim-dimensional vector.
        # padding_idx=0 leaves index 0 as an all-zero "null" embedding.
        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embed_dim,
            padding_idx=None,
        )

        # Each movie ID is similarly mapped to a dense vector.
        self.movie_embedding = nn.Embedding(
            num_embeddings=num_movies,
            embedding_dim=embed_dim,
            padding_idx=None,
        )

        # ------------------------------------------------------------------
        # MLP Tower  (Concatenation → Dense layers)
        # ------------------------------------------------------------------
        # Input size to the MLP = user_vec + movie_vec = 2 * embed_dim
        mlp_input_dim = embed_dim * 2

        layers = []
        in_features = mlp_input_dim

        for out_features in hidden_dims:
            layers.append(nn.Linear(in_features, out_features, bias=True))
            layers.append(nn.BatchNorm1d(out_features))   # stabilises training
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            in_features = out_features

        self.mlp = nn.Sequential(*layers)

        # ------------------------------------------------------------------
        # Output Layer  — predicts a single continuous rating value
        # ------------------------------------------------------------------
        self.output_layer = nn.Linear(in_features, 1, bias=True)

        # ------------------------------------------------------------------
        # Weight Initialisation — Xavier uniform for Linear, Normal for Embeddings
        # ------------------------------------------------------------------
        self._init_weights()

    # -----------------------------------------------------------------------

    def _init_weights(self):
        """Apply sensible default initialisations to all sub-modules."""
        # Embedding tables: small normal distribution prevents gradient saturation
        nn.init.normal_(self.user_embedding.weight,  mean=0.0, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, mean=0.0, std=0.01)

        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    # -----------------------------------------------------------------------

    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        user_ids  : LongTensor of shape (batch_size,)
        movie_ids : LongTensor of shape (batch_size,)

        Returns
        -------
        Tensor of shape (batch_size,) — predicted ratings in [rating_min, rating_max]
        """
        # ----- Embedding Lookup -----
        # user_vec  : (batch_size, embed_dim)
        # movie_vec : (batch_size, embed_dim)
        user_vec  = self.user_embedding(user_ids)
        movie_vec = self.movie_embedding(movie_ids)

        # ----- Concatenation -----
        # x : (batch_size, 2 * embed_dim)
        x = torch.cat([user_vec, movie_vec], dim=-1)

        # ----- MLP Tower -----
        # x : (batch_size, hidden_dims[-1])
        x = self.mlp(x)

        # ----- Output -----
        # raw : (batch_size, 1)
        raw = self.output_layer(x)

        # Squeeze to (batch_size,) and clamp to valid rating range
        prediction = torch.clamp(raw.squeeze(-1), self.rating_min, self.rating_max)

        return prediction

    # -----------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        total = self.count_parameters()
        return (
            super().__repr__()
            + f"\n\n[NCFModel] Trainable parameters: {total:,}"
        )


# ---------------------------------------------------------------------------
# 2B. NeuMF — combined GMF + MLP (extended architecture, optional)
# ---------------------------------------------------------------------------

class NeuMFModel(nn.Module):
    """
    Neural Matrix Factorisation (NeuMF) as proposed in He et al. (2017).

    Combines two complementary streams:
      * GMF  (Generalised Matrix Factorisation)  — element-wise product of embeddings
      * MLP  (Multi-Layer Perceptron)            — concatenated embeddings through dense layers

    Both streams share separate embedding tables for maximum expressiveness.
    Their final representations are concatenated and passed through a prediction layer.

    Parameters
    ----------
    num_users       : total unique users
    num_movies      : total unique movies
    gmf_embed_dim   : embedding size for the GMF stream (default 32)
    mlp_embed_dim   : embedding size for the MLP stream (default 32)
    mlp_hidden_dims : hidden layer sizes for the MLP stream (default [128, 64])
    dropout         : dropout probability (default 0.3)
    rating_min/max  : output clamping range
    """

    def __init__(
        self,
        num_users:       int,
        num_movies:      int,
        gmf_embed_dim:   int   = 32,
        mlp_embed_dim:   int   = 32,
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

        # ---- GMF Embeddings ----
        self.gmf_user_embedding  = nn.Embedding(num_users,  gmf_embed_dim)
        self.gmf_movie_embedding = nn.Embedding(num_movies, gmf_embed_dim)

        # ---- MLP Embeddings ----
        self.mlp_user_embedding  = nn.Embedding(num_users,  mlp_embed_dim)
        self.mlp_movie_embedding = nn.Embedding(num_movies, mlp_embed_dim)

        # ---- MLP Tower ----
        mlp_layers = []
        in_dim = mlp_embed_dim * 2
        for out_dim in mlp_hidden_dims:
            mlp_layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = out_dim
        self.mlp = nn.Sequential(*mlp_layers)

        # ---- Prediction Layer ----
        # Input = GMF output (gmf_embed_dim) + MLP final hidden (mlp_hidden_dims[-1])
        self.predict = nn.Linear(gmf_embed_dim + in_dim, 1, bias=True)

        self._init_weights()

    def _init_weights(self):
        for emb in [self.gmf_user_embedding, self.gmf_movie_embedding,
                    self.mlp_user_embedding, self.mlp_movie_embedding]:
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)

        for m in list(self.mlp.modules()) + [self.predict]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        # GMF stream: element-wise product of user and movie embeddings
        gmf_u = self.gmf_user_embedding(user_ids)
        gmf_m = self.gmf_movie_embedding(movie_ids)
        gmf_out = gmf_u * gmf_m                        # (batch, gmf_embed_dim)

        # MLP stream: concatenate then pass through tower
        mlp_u = self.mlp_user_embedding(user_ids)
        mlp_m = self.mlp_movie_embedding(movie_ids)
        mlp_x   = torch.cat([mlp_u, mlp_m], dim=-1)   # (batch, 2*mlp_embed_dim)
        mlp_out = self.mlp(mlp_x)                      # (batch, mlp_hidden_dims[-1])

        # Concatenate both streams
        combined = torch.cat([gmf_out, mlp_out], dim=-1)

        # Predict rating
        raw = self.predict(combined)
        return torch.clamp(raw.squeeze(-1), self.rating_min, self.rating_max)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 2C. Model factory — pick NCF or NeuMF by name
# ---------------------------------------------------------------------------

def build_model(
    model_type:  str,
    num_users:   int,
    num_movies:  int,
    **kwargs,
) -> nn.Module:
    """
    Factory function.

    Parameters
    ----------
    model_type : "ncf"  → NCFModel (pure MLP)
                 "neumf" → NeuMFModel (GMF + MLP)
    num_users  : total unique users
    num_movies : total unique movies
    **kwargs   : forwarded to the chosen model constructor

    Returns
    -------
    nn.Module
    """
    model_type = model_type.lower()

    if model_type == "ncf":
        model = NCFModel(num_users=num_users, num_movies=num_movies, **kwargs)
    elif model_type == "neumf":
        model = NeuMFModel(num_users=num_users, num_movies=num_movies, **kwargs)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'ncf' or 'neumf'.")

    print(f"\n[Model] Built '{model_type.upper()}'  "
          f"| Users: {num_users:,}  | Movies: {num_movies:,}  "
          f"| Params: {model.count_parameters():,}\n")
    print(model)
    return model

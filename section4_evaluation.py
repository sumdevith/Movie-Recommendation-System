"""
=============================================================================
SECTION 4: Distributed Model Evaluation
=============================================================================
Two complementary evaluation paths:

  PATH A — PyTorch native
    Load checkpoint → run model on test DataLoader → compute MSE / RMSE
    in Python. Fast, no Spark overhead, preferred for quick iteration.

  PATH B — PySpark distributed
    Load checkpoint → run inference in a Spark pandas_udf (vectorised UDF)
    → use RegressionEvaluator to compute RMSE across the full distributed
    test DataFrame. Preferred when the test set is truly massive or when
    you want evaluation integrated into a Spark pipeline.
=============================================================================
"""

import os
import math

import torch
import torch.nn as nn
import numpy as np

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, StructType, StructField
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd

from section2_model_architecture import build_model


# ---------------------------------------------------------------------------
# 4A. Load a saved checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: str, device: torch.device = None) -> dict:
    """
    Load a model checkpoint saved by the training loop.

    Parameters
    ----------
    checkpoint_path : path to the .pt file
    device          : target device (auto-detected if None)

    Returns
    -------
    dict with keys: model, num_users, num_movies, epoch, train_loss
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Eval] Loading checkpoint from '{checkpoint_path}' onto {device} …")
    ckpt = torch.load(checkpoint_path, map_location=device)

    model = build_model(
        model_type  = ckpt["model_type"],
        num_users   = ckpt["num_users"],
        num_movies  = ckpt["num_movies"],
        embed_dim   = ckpt["embed_dim"],
        hidden_dims = ckpt["hidden_dims"],
    ).to(device)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print(f"[Eval] Checkpoint loaded  |  epoch={ckpt['epoch']}  "
          f"|  train_loss={ckpt['train_loss']:.4f}")

    return {
        "model":      model,
        "num_users":  ckpt["num_users"],
        "num_movies": ckpt["num_movies"],
        "epoch":      ckpt["epoch"],
        "train_loss": ckpt["train_loss"],
        "device":     device,
    }


# ---------------------------------------------------------------------------
# 4B. PATH A — PyTorch-native evaluation (MSE + RMSE + MAE)
# ---------------------------------------------------------------------------

def evaluate_pytorch(
    model:       nn.Module,
    test_loader: "DataLoader",
    device:      torch.device,
) -> dict:
    """
    Evaluate the trained model on the test DataLoader using PyTorch.

    Metrics computed:
      MSE  — Mean Squared Error
      RMSE — Root Mean Squared Error  (primary metric for MovieLens benchmarks)
      MAE  — Mean Absolute Error

    Parameters
    ----------
    model       : loaded NCF / NeuMF nn.Module in eval mode
    test_loader : PyTorch DataLoader for the test partition
    device      : torch.device

    Returns
    -------
    dict  {mse, rmse, mae, num_samples}
    """
    model.eval()

    total_se   = 0.0     # sum of squared errors
    total_ae   = 0.0     # sum of absolute errors
    total_n    = 0

    print("[Eval] Running PyTorch evaluation on test set …")

    with torch.no_grad():
        for batch_idx, (user_ids, movie_ids, ratings) in enumerate(test_loader):
            user_ids  = user_ids.to(device, non_blocking=True)
            movie_ids = movie_ids.to(device, non_blocking=True)
            ratings   = ratings.to(device,  non_blocking=True)

            preds = model(user_ids, movie_ids)

            # Accumulate errors
            errors     = preds - ratings
            total_se  += (errors ** 2).sum().item()
            total_ae  += errors.abs().sum().item()
            total_n   += ratings.size(0)

            if (batch_idx + 1) % 50 == 0:
                running_rmse = (total_se / total_n) ** 0.5
                print(f"  Batch {batch_idx+1}/{len(test_loader)}  "
                      f"| Running RMSE: {running_rmse:.4f}")

    mse  = total_se / total_n
    rmse = mse ** 0.5
    mae  = total_ae / total_n

    print(f"\n{'='*55}")
    print(f"  PyTorch Evaluation Results  ({total_n:,} test samples)")
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}   ← primary benchmark metric")
    print(f"  MAE  : {mae:.4f}")
    print(f"{'='*55}\n")

    return {"mse": mse, "rmse": rmse, "mae": mae, "num_samples": total_n}


# ---------------------------------------------------------------------------
# 4C. PATH B — Spark-native distributed evaluation with pandas_udf
# ---------------------------------------------------------------------------

def evaluate_spark(
    test_df:         DataFrame,
    checkpoint_path: str,
    num_users:       int,
    num_movies:      int,
    model_type:      str  = "ncf",
    embed_dim:       int  = 64,
    hidden_dims:     list = None,
) -> dict:
    """
    Run inference inside a Spark pandas_udf so predictions are computed in
    parallel across all Spark workers, then evaluate with RegressionEvaluator.

    Architecture
    ────────────
    Each Spark partition → pandas_udf loads the checkpoint once per worker
    process (cached in a module-level dict to avoid redundant disk I/O) →
    runs forward pass on a CPU → returns a column of predictions →
    Spark aggregates all predictions → RegressionEvaluator computes RMSE.

    Parameters
    ----------
    test_df         : PySpark DataFrame with columns [user_idx, movie_idx, rating]
    checkpoint_path : absolute path to the .pt checkpoint file
    num_users       : total unique users (used if model is rebuilt)
    num_movies      : total unique movies
    model_type      : "ncf" or "neumf"
    embed_dim       : embedding dimension (must match checkpoint)
    hidden_dims     : MLP hidden layer sizes (must match checkpoint)

    Returns
    -------
    dict  {rmse, mse, num_samples}
    """
    if hidden_dims is None:
        hidden_dims = [256, 128, 64]

    spark = SparkSession.getActiveSession()

    # ---- Broadcast checkpoint bytes so every worker can reconstruct the model ----
    # Reading bytes lets us avoid a distributed filesystem requirement.
    with open(checkpoint_path, "rb") as f:
        ckpt_bytes = f.read()

    bc_ckpt_bytes    = spark.sparkContext.broadcast(ckpt_bytes)
    bc_num_users     = spark.sparkContext.broadcast(num_users)
    bc_num_movies    = spark.sparkContext.broadcast(num_movies)
    bc_model_type    = spark.sparkContext.broadcast(model_type)
    bc_embed_dim     = spark.sparkContext.broadcast(embed_dim)
    bc_hidden_dims   = spark.sparkContext.broadcast(hidden_dims)

    # ---- Define the pandas_udf ----
    # Pandas UDFs receive a pandas Series per column and return a Series.
    # We receive two Series (user_idx, movie_idx) and return predictions.

    # Module-level cache to avoid loading the model for every micro-batch
    _model_cache: dict = {}

    @F.pandas_udf(FloatType())
    def predict_udf(user_series: pd.Series, movie_series: pd.Series) -> pd.Series:
        import io
        import torch

        # Load model only once per worker process
        if "model" not in _model_cache:
            import section2_model_architecture as arch
            ckpt_io    = io.BytesIO(bc_ckpt_bytes.value)
            ckpt       = torch.load(ckpt_io, map_location="cpu")
            mdl = arch.build_model(
                model_type  = bc_model_type.value,
                num_users   = bc_num_users.value,
                num_movies  = bc_num_movies.value,
                embed_dim   = bc_embed_dim.value,
                hidden_dims = bc_hidden_dims.value,
            )
            mdl.load_state_dict(ckpt["state_dict"])
            mdl.eval()
            _model_cache["model"] = mdl

        model = _model_cache["model"]

        with torch.no_grad():
            u = torch.tensor(user_series.values,  dtype=torch.long)
            m = torch.tensor(movie_series.values, dtype=torch.long)
            p = model(u, m).numpy()

        return pd.Series(p.astype("float32"))

    # ---- Apply the UDF to the test DataFrame ----
    print("[Eval] Running distributed Spark inference …")
    predictions_df = test_df.withColumn(
        "prediction",
        predict_udf(F.col("user_idx"), F.col("movie_idx"))
    )

    # Cache so we don't recompute when calling .count() and evaluator
    predictions_df = predictions_df.cache()
    num_samples    = predictions_df.count()

    # ---- Spark RegressionEvaluator ----
    evaluator_rmse = RegressionEvaluator(
        predictionCol="prediction",
        labelCol="rating",
        metricName="rmse",
    )
    evaluator_mse = RegressionEvaluator(
        predictionCol="prediction",
        labelCol="rating",
        metricName="mse",
    )

    rmse = evaluator_rmse.evaluate(predictions_df)
    mse  = evaluator_mse.evaluate(predictions_df)

    print(f"\n{'='*55}")
    print(f"  Spark Distributed Evaluation Results  ({num_samples:,} test samples)")
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}   ← primary benchmark metric")
    print(f"{'='*55}\n")

    # Show a few sample predictions for sanity checking
    print("[Eval] Sample predictions vs. ground-truth ratings:")
    predictions_df.select("user_idx", "movie_idx", "rating", "prediction") \
                  .show(10, truncate=False)

    predictions_df.unpersist()

    return {"rmse": rmse, "mse": mse, "num_samples": num_samples}


# ---------------------------------------------------------------------------
# 4D. Full evaluation pipeline — runs both paths and prints a summary
# ---------------------------------------------------------------------------

def run_evaluation(
    checkpoint_path: str,
    test_loader:     "DataLoader",
    test_df:         DataFrame,
    use_spark_eval:  bool = True,
) -> dict:
    """
    Run both PyTorch and (optionally) Spark evaluations, then print a report.

    Parameters
    ----------
    checkpoint_path : path to best_model.pt
    test_loader     : PyTorch DataLoader for the test partition
    test_df         : PySpark test DataFrame (for Spark evaluation path)
    use_spark_eval  : whether to also run the Spark-native evaluation

    Returns
    -------
    dict with keys "pytorch" and (optionally) "spark"
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = load_checkpoint(checkpoint_path, device)

    results = {}

    # Path A — PyTorch
    pt_metrics = evaluate_pytorch(ckpt["model"], test_loader, device)
    results["pytorch"] = pt_metrics

    # Path B — Spark (optional, requires an active SparkSession)
    if use_spark_eval:
        spark = SparkSession.getActiveSession()
        if spark is not None:
            sk_metrics = evaluate_spark(
                test_df         = test_df,
                checkpoint_path = os.path.abspath(checkpoint_path),
                num_users       = ckpt["num_users"],
                num_movies      = ckpt["num_movies"],
            )
            results["spark"] = sk_metrics
        else:
            print("[Eval] No active SparkSession — skipping Spark evaluation.")

    # ---- Final summary ----
    print("\n" + "=" * 55)
    print("  FINAL EVALUATION SUMMARY")
    print("=" * 55)
    if "pytorch" in results:
        m = results["pytorch"]
        print(f"  [PyTorch]  RMSE={m['rmse']:.4f}  MSE={m['mse']:.4f}  MAE={m['mae']:.4f}")
    if "spark" in results:
        m = results["spark"]
        print(f"  [Spark]    RMSE={m['rmse']:.4f}  MSE={m['mse']:.4f}")
    print("=" * 55 + "\n")

    return results

"""
=============================================================================
SECTION 4: Distributed Model Evaluation
=============================================================================
Two complementary evaluation paths:

  PATH A — PyTorch native
    Load checkpoint → run model on test DataLoader → compute MSE / RMSE / MAE.

  PATH B — PySpark distributed
    Load checkpoint → run inference in a Spark pandas_udf (vectorised UDF)
    over the demographic + genre feature columns → use RegressionEvaluator
    to compute RMSE across the full distributed test DataFrame.
=============================================================================
"""

import os

import torch
import torch.nn as nn
import numpy as np

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd

from section2_model_architecture import build_model


# ---------------------------------------------------------------------------
# 4A. Load a saved checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: str, device: torch.device = None) -> dict:
    """Load a model checkpoint saved by the training loop."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Eval] Loading checkpoint from '{checkpoint_path}' onto {device} …")
    ckpt = torch.load(checkpoint_path, map_location=device)

    model = build_model(
        model_type   = ckpt["model_type"],
        feature_dims = ckpt["feature_dims"],
        embed_dim    = ckpt["embed_dim"],
        hidden_dims  = ckpt["hidden_dims"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print(f"[Eval] Checkpoint loaded  |  epoch={ckpt['epoch']}  |  train_loss={ckpt['train_loss']:.4f}")

    return {
        "model":        model,
        "feature_dims": ckpt["feature_dims"],
        "model_type":   ckpt["model_type"],
        "embed_dim":    ckpt["embed_dim"],
        "hidden_dims":  ckpt["hidden_dims"],
        "epoch":        ckpt["epoch"],
        "train_loss":   ckpt["train_loss"],
        "device":       device,
    }


# ---------------------------------------------------------------------------
# 4B. PATH A — PyTorch-native evaluation (MSE + RMSE + MAE)
# ---------------------------------------------------------------------------

def evaluate_pytorch(model: nn.Module, test_loader, device: torch.device) -> dict:
    """Evaluate the trained model on the test DataLoader using PyTorch."""
    model.eval()
    total_se = total_ae = 0.0
    total_n  = 0

    print("[Eval] Running PyTorch evaluation on test set …")
    with torch.no_grad():
        for batch_idx, (gender, age, occ, zip_region, genres, ratings) in enumerate(test_loader):
            gender     = gender.to(device, non_blocking=True)
            age        = age.to(device, non_blocking=True)
            occ        = occ.to(device, non_blocking=True)
            zip_region = zip_region.to(device, non_blocking=True)
            genres     = genres.to(device, non_blocking=True)
            ratings    = ratings.to(device, non_blocking=True)

            preds  = model(gender, age, occ, zip_region, genres)
            errors = preds - ratings
            total_se += (errors ** 2).sum().item()
            total_ae += errors.abs().sum().item()
            total_n  += ratings.size(0)

            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx+1}/{len(test_loader)}  "
                      f"| Running RMSE: {(total_se/total_n) ** 0.5:.4f}")

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
    feature_dims:    dict,
    model_type:      str  = "ncf",
    embed_dim:       int  = 32,
    hidden_dims:     list = None,
) -> dict:
    """Run inference inside a Spark pandas_udf (parallel across workers) then
    evaluate with RegressionEvaluator.

    ``test_df`` must carry the feature columns produced by section 1:
        gender_idx, age_idx, occ_idx, zip_idx, genres (array<float>), rating
    """
    if hidden_dims is None:
        hidden_dims = [128, 64]

    spark = SparkSession.getActiveSession()

    with open(checkpoint_path, "rb") as f:
        ckpt_bytes = f.read()

    bc_ckpt_bytes  = spark.sparkContext.broadcast(ckpt_bytes)
    bc_feature_dim = spark.sparkContext.broadcast(feature_dims)
    bc_model_type  = spark.sparkContext.broadcast(model_type)
    bc_embed_dim   = spark.sparkContext.broadcast(embed_dim)
    bc_hidden_dims = spark.sparkContext.broadcast(hidden_dims)

    _model_cache: dict = {}

    @F.pandas_udf(FloatType())
    def predict_udf(
        gender: pd.Series, age: pd.Series, occ: pd.Series,
        zip_region: pd.Series, genres: pd.Series,
    ) -> pd.Series:
        import io
        import numpy as np
        import torch

        if "model" not in _model_cache:
            import section2_model_architecture as arch
            ckpt = torch.load(io.BytesIO(bc_ckpt_bytes.value), map_location="cpu")
            mdl = arch.build_model(
                model_type   = bc_model_type.value,
                feature_dims = bc_feature_dim.value,
                embed_dim    = bc_embed_dim.value,
                hidden_dims  = bc_hidden_dims.value,
            )
            mdl.load_state_dict(ckpt["state_dict"])
            mdl.eval()
            _model_cache["model"] = mdl

        model = _model_cache["model"]

        with torch.no_grad():
            g  = torch.tensor(gender.values,     dtype=torch.long)
            a  = torch.tensor(age.values,        dtype=torch.long)
            o  = torch.tensor(occ.values,        dtype=torch.long)
            z  = torch.tensor(zip_region.values, dtype=torch.long)
            ge = torch.tensor(np.stack(genres.values), dtype=torch.float32)
            p  = model(g, a, o, z, ge).numpy()

        return pd.Series(p.astype("float32"))

    print("[Eval] Running distributed Spark inference …")
    predictions_df = test_df.withColumn(
        "prediction",
        predict_udf(
            F.col("gender_idx"), F.col("age_idx"), F.col("occ_idx"),
            F.col("zip_idx"), F.col("genres"),
        ),
    ).cache()

    num_samples = predictions_df.count()

    rmse = RegressionEvaluator(predictionCol="prediction", labelCol="rating", metricName="rmse").evaluate(predictions_df)
    mse  = RegressionEvaluator(predictionCol="prediction", labelCol="rating", metricName="mse").evaluate(predictions_df)

    print(f"\n{'='*55}")
    print(f"  Spark Distributed Evaluation Results  ({num_samples:,} test samples)")
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}   ← primary benchmark metric")
    print(f"{'='*55}\n")

    print("[Eval] Sample predictions vs. ground-truth ratings:")
    predictions_df.select("gender_idx", "age_idx", "occ_idx", "zip_idx", "rating", "prediction") \
                  .show(10, truncate=False)
    predictions_df.unpersist()

    return {"rmse": rmse, "mse": mse, "num_samples": num_samples}


# ---------------------------------------------------------------------------
# 4D. Full evaluation pipeline — runs both paths and prints a summary
# ---------------------------------------------------------------------------

def run_evaluation(checkpoint_path: str, test_loader, test_df: DataFrame, use_spark_eval: bool = True) -> dict:
    """Run both PyTorch and (optionally) Spark evaluations, then print a report."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = load_checkpoint(checkpoint_path, device)

    results = {"pytorch": evaluate_pytorch(ckpt["model"], test_loader, device)}

    if use_spark_eval:
        spark = SparkSession.getActiveSession()
        if spark is not None:
            results["spark"] = evaluate_spark(
                test_df         = test_df,
                checkpoint_path = os.path.abspath(checkpoint_path),
                feature_dims    = ckpt["feature_dims"],
                model_type      = ckpt["model_type"],
                embed_dim       = ckpt["embed_dim"],
                hidden_dims     = ckpt["hidden_dims"],
            )
        else:
            print("[Eval] No active SparkSession — skipping Spark evaluation.")

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

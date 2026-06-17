"""
=============================================================================
SECTION 5: End-to-End Execution Script
=============================================================================
Entry point that wires all four sections together into a single runnable
pipeline.

Usage (from the project root):
    uv run python main.py --data_path data/ratings.dat

All configurable hyper-parameters can be set via CLI flags; sensible
defaults are provided for a local development run on any laptop.
=============================================================================
"""

import os
import sys
import argparse
import time
import json

import torch

# ------------------------------------------------------------------
# Import all four pipeline sections
# ------------------------------------------------------------------
from section1_data_pipeline      import build_pipeline
from section2_model_architecture import build_model
from section3_distributed_training import launch_distributed_training
from section4_evaluation         import run_evaluation


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-End Distributed NCF Recommender System — MovieLens 1M"
    )

    # Data
    parser.add_argument(
        "--data_path", type=str, default="data/ratings.dat",
        help="Path to MovieLens 1M ratings.dat file"
    )

    # Model
    parser.add_argument(
        "--model_type", type=str, default="ncf",
        choices=["ncf", "neumf"],
        help="Model architecture: 'ncf' (pure MLP) or 'neumf' (GMF + MLP)"
    )
    parser.add_argument("--embed_dim",    type=int,   default=64,
                        help="Embedding dimension")
    parser.add_argument("--hidden_dims",  type=int,   nargs="+",
                        default=[256, 128, 64],
                        help="MLP hidden layer sizes (space-separated)")
    parser.add_argument("--dropout",      type=float, default=0.3,
                        help="Dropout probability")

    # Training
    parser.add_argument("--num_epochs",   type=int,   default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size",   type=int,   default=1024,
                        help="Mini-batch size")
    parser.add_argument("--lr",           type=float, default=1e-3,
                        help="Adam learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Adam L2 weight decay")

    # Distributed
    parser.add_argument("--num_workers",  type=int,   default=2,
                        help="Number of TorchDistributor worker processes")
    parser.add_argument("--use_gpu",      action="store_true",
                        help="Use NCCL GPU backend (requires CUDA)")

    # I/O
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--results_file",   type=str, default="results.json",
                        help="Path to save JSON evaluation results")

    # Evaluation
    parser.add_argument("--skip_spark_eval", action="store_true",
                        help="Skip the Spark distributed evaluation (PATH B)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    wall_start = time.time()

    print("\n" + "=" * 65)
    print("  MovieLens 1M — Distributed NCF Recommendation System")
    print("=" * 65)
    print(f"  Model       : {args.model_type.upper()}")
    print(f"  Embed dim   : {args.embed_dim}")
    print(f"  Hidden dims : {args.hidden_dims}")
    print(f"  Dropout     : {args.dropout}")
    print(f"  Epochs      : {args.num_epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  LR          : {args.lr}")
    print(f"  DDP workers : {args.num_workers}")
    print(f"  Use GPU     : {args.use_gpu}")
    print("=" * 65 + "\n")

    # ----------------------------------------------------------------
    # SECTION 1 — Distributed Data Pipeline
    # ----------------------------------------------------------------
    print("\n" + "─" * 65)
    print("  SECTION 1 : Distributed Data Pipeline (PySpark)")
    print("─" * 65)

    pipeline = build_pipeline(
        data_path  = args.data_path,
        batch_size = args.batch_size,
    )

    spark        = pipeline["spark"]
    train_loader = pipeline["train_loader"]
    test_loader  = pipeline["test_loader"]
    num_users    = pipeline["num_users"]
    num_movies   = pipeline["num_movies"]
    train_df     = pipeline["train_df"]
    test_df      = pipeline["test_df"]

    # ----------------------------------------------------------------
    # SECTION 2 — Architecture sanity check (dry run, no training)
    # ----------------------------------------------------------------
    print("\n" + "─" * 65)
    print("  SECTION 2 : Deep Learning Architecture")
    print("─" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Quick sanity-check forward pass
    sample_model = build_model(
        model_type  = args.model_type,
        num_users   = num_users,
        num_movies  = num_movies,
        embed_dim   = args.embed_dim,
        hidden_dims = args.hidden_dims,
        dropout     = args.dropout,
    ).to(device)

    dummy_u = torch.zeros(4, dtype=torch.long, device=device)
    dummy_m = torch.zeros(4, dtype=torch.long, device=device)
    dummy_p = sample_model(dummy_u, dummy_m)
    print(f"[Model] Sanity check forward pass → output shape: {tuple(dummy_p.shape)}  "
          f"values: {dummy_p.detach().cpu().numpy()}")
    del sample_model   # free memory before distributed launch

    # ----------------------------------------------------------------
    # SECTION 3 — Distributed Training
    # ----------------------------------------------------------------
    print("\n" + "─" * 65)
    print("  SECTION 3 : Distributed Training Loop (TorchDistributor)")
    print("─" * 65)

    train_result = launch_distributed_training(
        train_loader    = train_loader,
        num_users       = num_users,
        num_movies      = num_movies,
        num_workers     = args.num_workers,
        use_gpu         = args.use_gpu,
        # Model kwargs
        model_type      = args.model_type,
        embed_dim       = args.embed_dim,
        hidden_dims     = args.hidden_dims,
        dropout         = args.dropout,
        # Training kwargs
        num_epochs      = args.num_epochs,
        batch_size      = args.batch_size,
        lr              = args.lr,
        weight_decay    = args.weight_decay,
        checkpoint_dir  = args.checkpoint_dir,
    )

    checkpoint_path = train_result["best_checkpoint"]
    print(f"\n[Main] Best checkpoint: {checkpoint_path}")

    # ----------------------------------------------------------------
    # SECTION 4 — Evaluation
    # ----------------------------------------------------------------
    print("\n" + "─" * 65)
    print("  SECTION 4 : Distributed Model Evaluation")
    print("─" * 65)

    eval_results = run_evaluation(
        checkpoint_path = checkpoint_path,
        test_loader     = test_loader,
        test_df         = test_df,
        use_spark_eval  = not args.skip_spark_eval,
    )

    # ----------------------------------------------------------------
    # Save results to JSON
    # ----------------------------------------------------------------
    output = {
        "config": vars(args),
        "num_users":   num_users,
        "num_movies":  num_movies,
        "epoch_losses": train_result.get("epoch_losses", []),
        "evaluation":   eval_results,
    }

    # Convert any non-serialisable values
    for k, v in output["evaluation"].items():
        for mk, mv in v.items():
            if hasattr(mv, "item"):           # numpy / torch scalar
                output["evaluation"][k][mk] = float(mv)

    with open(args.results_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[Main] Results written to '{args.results_file}'")

    # ----------------------------------------------------------------
    # Housekeeping
    # ----------------------------------------------------------------
    spark.stop()
    print(f"\n[Main] SparkSession stopped.")

    wall_elapsed = time.time() - wall_start
    print(f"[Main] Total wall-clock time: {wall_elapsed/60:.1f} minutes\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

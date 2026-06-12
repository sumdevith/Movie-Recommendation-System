"""
=============================================================================
SECTION 3: Distributed Training Loop
=============================================================================
Strategy: PyTorch DistributedDataParallel (DDP) launched via
          pyspark.ml.torch.distributor.TorchDistributor

TorchDistributor:
  • Launches one PyTorch worker process per Spark executor
  • Each worker receives the full training function + its rank (local_rank)
  • Gradients are averaged across all workers via NCCL (GPU) or Gloo (CPU)
  • Worker 0 is the "master" that logs metrics and saves checkpoints

Fallback:
  If TorchDistributor is unavailable (Spark < 3.4) we fall back to
  single-process training with DataParallel on multi-GPU machines.
=============================================================================
"""

import os
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from section2_model_architecture import build_model
from section1_data_pipeline import RatingsDataset


# ---------------------------------------------------------------------------
# 3A. Single-worker training step (called inside distributed worker)
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  nn.Module,
    optimizer:  optim.Optimizer,
    device:     torch.device,
    epoch:      int,
    rank:       int = 0,
) -> float:
    """
    Run one full pass over the training data on `device`.

    Parameters
    ----------
    model     : the NCF / NeuMF nn.Module (possibly wrapped in DDP)
    loader    : DataLoader for this worker's data shard
    criterion : loss function (MSELoss)
    optimizer : Adam
    device    : torch.device('cuda:N') or torch.device('cpu')
    epoch     : current epoch index (0-based) — used only for logging
    rank      : distributed rank — worker 0 prints progress

    Returns
    -------
    mean_loss : average MSE loss across all batches in this epoch
    """
    model.train()
    running_loss  = 0.0
    total_batches = len(loader)

    for batch_idx, (user_ids, movie_ids, ratings) in enumerate(loader):
        # Move tensors to the target device
        user_ids  = user_ids.to(device, non_blocking=True)
        movie_ids = movie_ids.to(device, non_blocking=True)
        ratings   = ratings.to(device, non_blocking=True)

        # ---- Forward pass ----
        optimizer.zero_grad(set_to_none=True)    # slightly faster than zero_grad()
        predictions = model(user_ids, movie_ids)

        # ---- Loss (MSE) ----
        loss = criterion(predictions, ratings)

        # ---- Backward pass ----
        loss.backward()

        # Gradient clipping — prevents exploding gradients in deep networks
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()

        # Print batch progress every 200 batches (only rank 0)
        if rank == 0 and (batch_idx + 1) % 200 == 0:
            print(
                f"  [Epoch {epoch+1}] Batch {batch_idx+1}/{total_batches}  "
                f"| Batch MSE: {loss.item():.4f}"
            )

    mean_loss = running_loss / total_batches
    return mean_loss


# ---------------------------------------------------------------------------
# 3B. Distributed training function — executed inside each worker process
# ---------------------------------------------------------------------------

def distributed_train_fn(
    # ---- Dataset arrays (serialised by TorchDistributor via pickle) ----
    train_user_idx:  "torch.Tensor",
    train_movie_idx: "torch.Tensor",
    train_ratings:   "torch.Tensor",
    # ---- Model hyper-parameters ----
    num_users:    int,
    num_movies:   int,
    model_type:   str   = "ncf",
    embed_dim:    int   = 64,
    hidden_dims:  list  = None,
    dropout:      float = 0.3,
    # ---- Training hyper-parameters ----
    num_epochs:   int   = 10,
    batch_size:   int   = 1024,
    lr:           float = 1e-3,
    weight_decay: float = 1e-5,
    # ---- Checkpoint ----
    checkpoint_dir: str = "checkpoints",
):
    """
    Core training function executed by every distributed worker.

    TorchDistributor sets these env vars before calling this function:
      LOCAL_RANK  — GPU / CPU index on this node
      RANK        — global worker index
      WORLD_SIZE  — total number of workers

    Returns (from rank 0 only):
      dict with epoch_losses list and best checkpoint path
    """
    if hidden_dims is None:
        hidden_dims = [256, 128, 64]

    # ------------------------------------------------------------------
    # Distributed process group initialisation
    # ------------------------------------------------------------------
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK",       0))
    world_size  = int(os.environ.get("WORLD_SIZE",  1))

    # Choose backend: NCCL for GPU (fastest), Gloo for CPU
    if torch.cuda.is_available():
        device  = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    else:
        device  = torch.device("cpu")
        backend = "gloo"

    # Initialise the distributed process group (no-op if world_size == 1)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)

    if global_rank == 0:
        print(f"\n[Distributed] world_size={world_size}  backend={backend}  device={device}")

    # ------------------------------------------------------------------
    # Build Dataset & DistributedSampler
    # ------------------------------------------------------------------
    # Re-wrap the raw tensors into a TensorDataset for the DataLoader
    from torch.utils.data import TensorDataset
    full_dataset = TensorDataset(train_user_idx, train_movie_idx, train_ratings)

    # DistributedSampler ensures each worker sees a non-overlapping shard
    sampler = DistributedSampler(
        full_dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True,
        seed=42,
    ) if world_size > 1 else None

    train_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),   # only shuffle when no DistributedSampler
        num_workers=2,
        pin_memory=(backend == "nccl"),
    )

    # ------------------------------------------------------------------
    # Build Model  →  move to device  →  wrap in DDP
    # ------------------------------------------------------------------
    model = build_model(
        model_type  = model_type,
        num_users   = num_users,
        num_movies  = num_movies,
        embed_dim   = embed_dim,
        hidden_dims = hidden_dims,
        dropout     = dropout,
    ).to(device)

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank] if backend == "nccl" else None,
        )

    # ------------------------------------------------------------------
    # Loss function and Optimiser
    # ------------------------------------------------------------------
    criterion = nn.MSELoss()                # Mean Squared Error — standard for rating regression
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,          # L2 regularisation
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Cosine annealing LR scheduler — smoothly reduces LR from lr → 0
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01
    )

    # ------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------
    epoch_losses = []
    best_loss    = math.inf
    best_ckpt    = None

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Tell DistributedSampler which epoch we are on → different shuffle each epoch
        if sampler is not None:
            sampler.set_epoch(epoch)

        mean_loss = train_one_epoch(
            model     = model,
            loader    = train_loader,
            criterion = criterion,
            optimizer = optimizer,
            device    = device,
            epoch     = epoch,
            rank      = global_rank,
        )

        scheduler.step()
        epoch_losses.append(mean_loss)

        elapsed = time.time() - epoch_start

        # ---- Aggregate loss across all workers (all-reduce mean) ----
        if world_size > 1:
            loss_tensor = torch.tensor(mean_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            mean_loss = loss_tensor.item()

        # ---- Logging & checkpointing (rank 0 only) ----
        if global_rank == 0:
            rmse_approx = mean_loss ** 0.5
            current_lr  = scheduler.get_last_lr()[0]
            print(
                f"[Epoch {epoch+1:>3}/{num_epochs}]  "
                f"Train MSE: {mean_loss:.4f}  "
                f"Train RMSE ≈ {rmse_approx:.4f}  "
                f"LR: {current_lr:.2e}  "
                f"Time: {elapsed:.1f}s"
            )

            # Save best model checkpoint
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_ckpt = os.path.join(checkpoint_dir, "best_model.pt")
                # Unwrap DDP before saving
                state_dict = (
                    model.module.state_dict()
                    if isinstance(model, DDP)
                    else model.state_dict()
                )
                torch.save(
                    {
                        "epoch":      epoch + 1,
                        "state_dict": state_dict,
                        "optimizer":  optimizer.state_dict(),
                        "train_loss": mean_loss,
                        "num_users":  num_users,
                        "num_movies": num_movies,
                        "model_type": model_type,
                        "embed_dim":  embed_dim,
                        "hidden_dims": hidden_dims,
                    },
                    best_ckpt,
                )
                print(f"  ✔ Checkpoint saved → {best_ckpt}  (loss={best_loss:.4f})")

    # ------------------------------------------------------------------
    # Clean up process group
    # ------------------------------------------------------------------
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()

    if global_rank == 0:
        print(f"\n[Training Complete]  Best Train MSE: {best_loss:.4f}  "
              f"| Checkpoint: {best_ckpt}")
        return {"epoch_losses": epoch_losses, "best_checkpoint": best_ckpt}

    return None   # non-zero ranks return nothing


# ---------------------------------------------------------------------------
# 3C. Launcher — wraps distributed_train_fn with TorchDistributor
# ---------------------------------------------------------------------------

def launch_distributed_training(
    train_loader: DataLoader,
    num_users:    int,
    num_movies:   int,
    num_workers:  int  = 2,
    use_gpu:      bool = False,
    **train_kwargs,
) -> dict:
    """
    Launch distributed training via TorchDistributor (Spark ≥ 3.4) with
    an automatic fallback to single-process training for older Spark versions.

    Parameters
    ----------
    train_loader : PyTorch DataLoader (used to extract the raw tensors)
    num_users    : total unique users
    num_movies   : total unique movies
    num_workers  : number of parallel Spark worker processes
    use_gpu      : whether to use NCCL + CUDA (set True on GPU clusters)
    **train_kwargs : forwarded to distributed_train_fn

    Returns
    -------
    dict from rank-0 worker: {epoch_losses, best_checkpoint}
    """
    # Materialise the full dataset from the DataLoader so we can pass tensors
    # to TorchDistributor (which serialises arguments via pickle)
    print("[Launcher] Collecting training tensors for distribution …")
    all_users, all_movies, all_ratings = [], [], []
    for u, m, r in train_loader:
        all_users.append(u)
        all_movies.append(m)
        all_ratings.append(r)

    train_user_idx  = torch.cat(all_users)
    train_movie_idx = torch.cat(all_movies)
    train_ratings   = torch.cat(all_ratings)

    print(f"[Launcher] Tensor shapes  "
          f"users={tuple(train_user_idx.shape)}  "
          f"movies={tuple(train_movie_idx.shape)}  "
          f"ratings={tuple(train_ratings.shape)}")

    # Attempt TorchDistributor (PySpark ≥ 3.4)
    try:
        from pyspark.ml.torch.distributor import TorchDistributor
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError("No active SparkSession found.")

        print(f"[Launcher] Starting TorchDistributor  "
              f"num_processes={num_workers}  use_gpu={use_gpu}")

        distributor = TorchDistributor(
            num_processes=num_workers,
            local_mode=True,            # all workers on this single machine
            use_gpu=use_gpu,
        )

        result = distributor.run(
            distributed_train_fn,
            # Positional tensor arguments
            train_user_idx,
            train_movie_idx,
            train_ratings,
            # Keyword arguments
            num_users   = num_users,
            num_movies  = num_movies,
            **train_kwargs,
        )

    except ImportError:
        # PySpark < 3.4  or  TorchDistributor not installed
        print("[Launcher] TorchDistributor not available — "
              "falling back to single-process training.")

        # Emulate single-worker environment variables
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("RANK",       "0")
        os.environ.setdefault("WORLD_SIZE", "1")

        result = distributed_train_fn(
            train_user_idx,
            train_movie_idx,
            train_ratings,
            num_users  = num_users,
            num_movies = num_movies,
            **train_kwargs,
        )

    return result

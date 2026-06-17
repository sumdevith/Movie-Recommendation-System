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
  single-process training.

The model is the demographic + genre recommender, so every batch carries the
five feature tensors (gender, age, occupation, zip region, genre multi-hot).
=============================================================================
"""

import os
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from section2_model_architecture import build_model


# ---------------------------------------------------------------------------
# 3A. Single-worker training step
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, rank=0):
    """Run one full pass over the training data on ``device``."""
    model.train()
    running_loss  = 0.0
    total_batches = len(loader)

    for batch_idx, (gender, age, occ, zip_region, genres, ratings) in enumerate(loader):
        gender     = gender.to(device, non_blocking=True)
        age        = age.to(device, non_blocking=True)
        occ        = occ.to(device, non_blocking=True)
        zip_region = zip_region.to(device, non_blocking=True)
        genres     = genres.to(device, non_blocking=True)
        ratings    = ratings.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(gender, age, occ, zip_region, genres)
        loss = criterion(predictions, ratings)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

        if rank == 0 and (batch_idx + 1) % 200 == 0:
            print(f"  [Epoch {epoch+1}] Batch {batch_idx+1}/{total_batches}  "
                  f"| Batch MSE: {loss.item():.4f}")

    return running_loss / total_batches


# ---------------------------------------------------------------------------
# 3B. Distributed training function — executed inside each worker process
# ---------------------------------------------------------------------------

def distributed_train_fn(
    # ---- Dataset feature tensors (serialised by TorchDistributor) ----
    train_gender:  "torch.Tensor",
    train_age:     "torch.Tensor",
    train_occ:     "torch.Tensor",
    train_zip:     "torch.Tensor",
    train_genres:  "torch.Tensor",
    train_ratings: "torch.Tensor",
    # ---- Model hyper-parameters ----
    feature_dims:  dict,
    model_type:    str   = "ncf",
    embed_dim:     int   = 32,
    hidden_dims:   list  = None,
    dropout:       float = 0.3,
    # ---- Training hyper-parameters ----
    num_epochs:    int   = 10,
    batch_size:    int   = 1024,
    lr:            float = 1e-3,
    weight_decay:  float = 1e-5,
    # ---- Checkpoint ----
    checkpoint_dir: str = "checkpoints",
):
    """Core training function executed by every distributed worker."""
    if hidden_dims is None:
        hidden_dims = [128, 64]

    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK",       0))
    world_size  = int(os.environ.get("WORLD_SIZE",  1))

    if torch.cuda.is_available():
        device  = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    else:
        device  = torch.device("cpu")
        backend = "gloo"

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)

    if global_rank == 0:
        print(f"\n[Distributed] world_size={world_size}  backend={backend}  device={device}")

    # ------------------------------------------------------------------
    # Build Dataset & DistributedSampler
    # ------------------------------------------------------------------
    full_dataset = TensorDataset(
        train_gender, train_age, train_occ, train_zip, train_genres, train_ratings
    )

    sampler = DistributedSampler(
        full_dataset, num_replicas=world_size, rank=global_rank,
        shuffle=True, seed=42,
    ) if world_size > 1 else None

    train_loader = DataLoader(
        full_dataset, batch_size=batch_size, sampler=sampler,
        shuffle=(sampler is None), num_workers=2, pin_memory=(backend == "nccl"),
    )

    # ------------------------------------------------------------------
    # Build Model → device → DDP
    # ------------------------------------------------------------------
    model = build_model(
        model_type   = model_type,
        feature_dims = feature_dims,
        embed_dim    = embed_dim,
        hidden_dims  = hidden_dims,
        dropout      = dropout,
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if backend == "nccl" else None)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay,
        betas=(0.9, 0.999), eps=1e-8,
    )
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
        if sampler is not None:
            sampler.set_epoch(epoch)

        mean_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, global_rank
        )
        scheduler.step()
        epoch_losses.append(mean_loss)
        elapsed = time.time() - epoch_start

        if world_size > 1:
            loss_tensor = torch.tensor(mean_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            mean_loss = loss_tensor.item()

        if global_rank == 0:
            print(f"[Epoch {epoch+1:>3}/{num_epochs}]  "
                  f"Train MSE: {mean_loss:.4f}  Train RMSE ≈ {mean_loss**0.5:.4f}  "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}  Time: {elapsed:.1f}s")

            if mean_loss < best_loss:
                best_loss = mean_loss
                best_ckpt = os.path.join(checkpoint_dir, "best_model.pt")
                state_dict = (model.module.state_dict()
                              if isinstance(model, DDP) else model.state_dict())
                torch.save(
                    {
                        "epoch":        epoch + 1,
                        "state_dict":   state_dict,
                        "optimizer":    optimizer.state_dict(),
                        "train_loss":   mean_loss,
                        "feature_dims": feature_dims,
                        "model_type":   model_type,
                        "embed_dim":    embed_dim,
                        "hidden_dims":  hidden_dims,
                    },
                    best_ckpt,
                )
                print(f"  ✔ Checkpoint saved → {best_ckpt}  (loss={best_loss:.4f})")

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()

    if global_rank == 0:
        print(f"\n[Training Complete]  Best Train MSE: {best_loss:.4f}  | Checkpoint: {best_ckpt}")
        return {"epoch_losses": epoch_losses, "best_checkpoint": best_ckpt}

    return None


# ---------------------------------------------------------------------------
# 3C. Launcher — wraps distributed_train_fn with TorchDistributor
# ---------------------------------------------------------------------------

def launch_distributed_training(
    train_loader: DataLoader,
    feature_dims: dict,
    num_workers:  int  = 2,
    use_gpu:      bool = False,
    **train_kwargs,
) -> dict:
    """Launch distributed training via TorchDistributor (Spark ≥ 3.4) with an
    automatic fallback to single-process training for older Spark versions."""
    print("[Launcher] Collecting training tensors for distribution …")
    g, a, o, z, ge, r = [], [], [], [], [], []
    for gender, age, occ, zip_region, genres, ratings in train_loader:
        g.append(gender); a.append(age); o.append(occ)
        z.append(zip_region); ge.append(genres); r.append(ratings)

    train_gender  = torch.cat(g)
    train_age     = torch.cat(a)
    train_occ     = torch.cat(o)
    train_zip     = torch.cat(z)
    train_genres  = torch.cat(ge)
    train_ratings = torch.cat(r)

    print(f"[Launcher] Tensor shapes  gender={tuple(train_gender.shape)}  "
          f"genres={tuple(train_genres.shape)}  ratings={tuple(train_ratings.shape)}")

    tensor_args = (train_gender, train_age, train_occ, train_zip, train_genres, train_ratings)

    try:
        from pyspark.ml.torch.distributor import TorchDistributor
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError("No active SparkSession found.")

        print(f"[Launcher] Starting TorchDistributor  num_processes={num_workers}  use_gpu={use_gpu}")
        distributor = TorchDistributor(
            num_processes=num_workers, local_mode=True, use_gpu=use_gpu,
        )
        result = distributor.run(
            distributed_train_fn,
            *tensor_args,
            feature_dims=feature_dims,
            **train_kwargs,
        )

    except ImportError:
        print("[Launcher] TorchDistributor not available — falling back to single-process training.")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("RANK",       "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        result = distributed_train_fn(
            *tensor_args,
            feature_dims=feature_dims,
            **train_kwargs,
        )

    return result

# Distributed NCF Recommender System - MovieLens 1M

A production-grade, end-to-end **Neural Collaborative Filtering (NCF)** movie
recommendation system built on:

| Layer | Technology |
|---|---|
| Distributed data pipeline | **PySpark 3.4+** |
| Deep learning model | **PyTorch** (NCF / NeuMF) |
| Distributed training | **TorchDistributor** (DDP) |
| Distributed evaluation | **Spark RegressionEvaluator** + pandas_udf |

---

## Project Structure

```text
ncf_recommender/
|-- section1_data_pipeline.py        # PySpark data loading, ID remapping, DataLoaders
|-- section2_model_architecture.py   # NCFModel + NeuMFModel PyTorch classes
|-- section3_distributed_training.py # DDP training loop + TorchDistributor launcher
|-- section4_evaluation.py           # PyTorch + Spark evaluation (RMSE, MSE, MAE)
|-- main.py                          # End-to-end CLI entry point
|-- ncf_movielens.ipynb              # Jupyter Notebook version
|-- pyproject.toml
`-- data/                            # ratings.dat placed here
```

---

## Quick Start

### Notebook

```bash
# 1. Clone / copy the project files
# 2. Create a local .venv with notebook dependencies
uv sync --group notebook

# 3. Open the notebook
uv run jupyter lab ncf_movielens.ipynb
```

The notebook expects the manually extracted MovieLens 1M ratings file at
`data/ratings.dat`.

### CLI

```bash
# Create venv and install deps
uv sync

# Run with default settings (NCF, 10 epochs, 2 DDP workers)
uv run python main.py --data_path data/ratings.dat

# Run NeuMF with more epochs
uv run python main.py \
  --data_path     data/ratings.dat \
  --model_type    neumf \
  --embed_dim     64 \
  --hidden_dims   256 128 64 \
  --num_epochs    20 \
  --batch_size    2048 \
  --num_workers   4
```

Both the notebook and CLI expect `data/ratings.dat` to exist before running.

---

## Architecture: NCFModel (pure MLP)

```text
User ID  -> Embedding(num_users,  64) -> u_vec (64-d)
Movie ID -> Embedding(num_movies, 64) -> m_vec (64-d)

concat([u_vec, m_vec]) -> x (128-d)
  |
  |-- Linear(128 -> 256) + BatchNorm + ReLU + Dropout(0.3)
  |-- Linear(256 -> 128) + BatchNorm + ReLU + Dropout(0.3)
  |-- Linear(128 ->  64) + BatchNorm + ReLU + Dropout(0.3)
  `-- Linear( 64 ->   1) + clamp(1, 5)

Loss: MSELoss   Optimiser: Adam   Scheduler: CosineAnnealingLR
```

---

## Architecture: NeuMFModel (GMF + MLP)

```text
GMF stream:  u_gmf * m_gmf      -> gmf_out (32-d element-wise product)
MLP stream:  concat(u_mlp, m_mlp) -> tower -> mlp_out (64-d)

concat([gmf_out, mlp_out]) -> Linear(96 -> 1) -> clamp(1, 5)
```

---

## Distributed Strategy

| Step | Technology | Detail |
|---|---|---|
| Data loading | `PySpark.randomSplit` | 80/20 split across all Spark partitions |
| ID remapping | `PySpark broadcast UDF` | Maps sparse IDs to contiguous 0-based indices |
| DataLoader | `DistributedSampler` | Each DDP rank sees a non-overlapping data shard |
| Training | `TorchDistributor` | Spawns N worker processes; gradients averaged via Gloo/NCCL |
| Checkpointing | Worker rank 0 | Saves best model to `checkpoints/best_model.pt` |
| Evaluation | `pandas_udf` + `RegressionEvaluator` | Inference runs in parallel on each Spark partition |

---

## Expected Results (MovieLens 1M)

| Model | RMSE (test) | Epochs |
|---|---|---|
| NCF (64-d, [256,128,64]) | ~0.87-0.90 | 10 |
| NeuMF (32+32-d)          | ~0.85-0.88 | 15 |

> Standard MF baseline: ~0.91 RMSE. NCF/NeuMF consistently beat it.

---

## Requirements

- Python >= 3.9
- uv
- Java >= 8 (required by PySpark)
- PySpark >= 3.4 (for TorchDistributor)
- PyTorch >= 2.1

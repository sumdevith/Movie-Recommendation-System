# Distributed NCF Recommender System — MovieLens 1M

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

```
ncf_recommender/
├── section1_data_pipeline.py       # PySpark data loading, ID remapping, DataLoaders
├── section2_model_architecture.py  # NCFModel + NeuMFModel PyTorch classes
├── section3_distributed_training.py# DDP training loop + TorchDistributor launcher
├── section4_evaluation.py          # PyTorch + Spark evaluation (RMSE, MSE, MAE)
├── main.py                         # End-to-end CLI entry point
├── ncf_movielens.ipynb             # Jupyter Notebook version
├── requirements.txt
├── run_ncf.sh                      # One-shot setup + run script
└── data/                           # ratings.dat placed here
```

---

## Quick Start

```bash
# 1. Clone / copy the project files
# 2. Run the all-in-one setup + training script
chmod +x run_ncf.sh
./run_ncf.sh
```

The script will:
1. Create a Python venv and install all dependencies
2. Download + extract `ml-1m.zip` from GroupLens
3. Launch the full pipeline (PySpark → NCF → TorchDistributor → Evaluation)

---

## Manual Run

```bash
# Create venv
python3 -m venv .venv && source .venv/bin/activate

# Install deps
pip install -r requirements.txt

# Download dataset
curl -L -o data/ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip data/ml-1m.zip -d data/
mv data/ml-1m/ratings.dat data/

# Run with default settings (NCF, 10 epochs, 2 DDP workers)
python main.py --data_path data/ratings.dat

# Run NeuMF with more epochs
python main.py \
  --data_path     data/ratings.dat \
  --model_type    neumf \
  --embed_dim     64 \
  --hidden_dims   256 128 64 \
  --num_epochs    20 \
  --batch_size    2048 \
  --num_workers   4
```

---

## Architecture: NCFModel (pure MLP)

```
User ID  ──► Embedding(num_users,  64)  ──► u_vec  (64-d)
Movie ID ──► Embedding(num_movies, 64)  ──► m_vec  (64-d)

concat([u_vec, m_vec])  →  x  (128-d)
  │
  ├─ Linear(128 → 256) + BatchNorm + ReLU + Dropout(0.3)
  ├─ Linear(256 → 128) + BatchNorm + ReLU + Dropout(0.3)
  ├─ Linear(128 →  64) + BatchNorm + ReLU + Dropout(0.3)
  └─ Linear( 64 →   1) + clamp(1, 5)

Loss: MSELoss   Optimiser: Adam   Scheduler: CosineAnnealingLR
```

---

## Architecture: NeuMFModel (GMF + MLP)

```
GMF stream:  u_gmf ⊙ m_gmf  →  gmf_out  (32-d element-wise product)
MLP stream:  concat(u_mlp, m_mlp)  →  tower  →  mlp_out  (64-d)

concat([gmf_out, mlp_out])  →  Linear(96 → 1)  →  clamp(1, 5)
```

---

## Distributed Strategy

| Step | Technology | Detail |
|---|---|---|
| Data loading | `PySpark.randomSplit` | 80/20 split across all Spark partitions |
| ID remapping | `PySpark broadcast UDF` | Maps sparse IDs → contiguous 0-based indices |
| DataLoader | `DistributedSampler` | Each DDP rank sees a non-overlapping data shard |
| Training | `TorchDistributor` | Spawns N worker processes; gradients averaged via Gloo/NCCL |
| Checkpointing | Worker rank 0 | Saves best model to `checkpoints/best_model.pt` |
| Evaluation | `pandas_udf` + `RegressionEvaluator` | Inference runs in parallel on each Spark partition |

---

## Expected Results (MovieLens 1M)

| Model | RMSE (test) | Epochs |
|---|---|---|
| NCF (64-d, [256,128,64]) | ~0.87–0.90 | 10 |
| NeuMF (32+32-d)          | ~0.85–0.88 | 15 |

> Standard MF baseline: ~0.91 RMSE. NCF/NeuMF consistently beat it.

---

## Requirements

- Python ≥ 3.9
- Java ≥ 8 (required by PySpark)
- PySpark ≥ 3.4 (for TorchDistributor)
- PyTorch ≥ 2.1

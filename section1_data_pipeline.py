"""
=============================================================================
SECTION 1: Distributed Data Pipeline (PySpark)
=============================================================================
MovieLens 1M Dataset — Neural Collaborative Filtering
Pipeline: Load → Clean → Split → Convert to PyTorch DataLoaders

Dataset: https://files.grouplens.org/datasets/movielens/ml-1m.zip
File used: ratings.dat  (1,000,209 rows)
Format:    UserID::MovieID::Rating::Timestamp
=============================================================================
"""

import os
import multiprocessing

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, IntegerType, FloatType, LongType
)


# ---------------------------------------------------------------------------
# 1A. SparkSession — maximise all available cores, tune memory
# ---------------------------------------------------------------------------

def create_spark_session(app_name: str = "NCF_MovieLens") -> SparkSession:
    """
    Initialise a local SparkSession that uses every available CPU core.

    Memory knobs:
      - driver.memory        : heap for the driver JVM (data collection, etc.)
      - executor.memory      : heap per executor JVM
      - sql.shuffle.partitions: reduces overhead for datasets that fit in RAM

    Returns
    -------
    SparkSession
    """
    num_cores = multiprocessing.cpu_count()
    master    = f"local[{num_cores}]"          # use ALL logical cores

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        # ----- memory -----
        .config("spark.driver.memory",              "4g")
        .config("spark.executor.memory",            "4g")
        .config("spark.driver.maxResultSize",       "2g")
        # ----- shuffle -----
        .config("spark.sql.shuffle.partitions",     str(num_cores * 2))
        # ----- serialisation (faster for ML workloads) -----
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        # ----- UI (disable to keep output clean in notebooks) -----
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")        # suppress INFO noise

    print(f"[Spark] Session started  |  master={master}  |  cores={num_cores}")
    print(f"[Spark] Spark version: {spark.version}")
    return spark


# ---------------------------------------------------------------------------
# 1B. Load ratings.dat  (UserID::MovieID::Rating::Timestamp)
# ---------------------------------------------------------------------------

def load_ratings(spark: SparkSession, data_path: str):
    """
    Load MovieLens 1M ratings file using the '::' separator.

    Parameters
    ----------
    spark     : active SparkSession
    data_path : path to ratings.dat

    Returns
    -------
    pyspark.sql.DataFrame  with columns [user_id, movie_id, rating, timestamp]
    """
    # Explicit schema — avoids an expensive inference scan
    schema = StructType([
        StructField("user_id",   IntegerType(), nullable=False),
        StructField("movie_id",  IntegerType(), nullable=False),
        StructField("rating",    FloatType(),   nullable=False),
        StructField("timestamp", LongType(),    nullable=False),
    ])

    # ratings.dat uses '::' which Spark's CSV reader does not support natively.
    # We read as raw text then split manually — reliable across all Spark versions.
    raw_df = spark.read.text(data_path)

    ratings_df = (
        raw_df
        .select(F.split(F.col("value"), "::").alias("parts"))
        .select(
            F.col("parts")[0].cast(IntegerType()).alias("user_id"),
            F.col("parts")[1].cast(IntegerType()).alias("movie_id"),
            F.col("parts")[2].cast(FloatType()).alias("rating"),
            F.col("parts")[3].cast(LongType()).alias("timestamp"),
        )
        # Drop any row that failed to parse
        .dropna(subset=["user_id", "movie_id", "rating"])
    )

    total = ratings_df.count()
    print(f"[Data] Loaded {total:,} ratings from '{data_path}'")
    ratings_df.show(5, truncate=False)
    ratings_df.printSchema()
    return ratings_df


# ---------------------------------------------------------------------------
# 1C. Build contiguous 0-based ID maps (required by embedding layers)
# ---------------------------------------------------------------------------

def build_id_maps(ratings_df):
    """
    Embedding layers require contiguous integer IDs starting at 0.
    MovieLens user IDs are 1-6040 and movie IDs are sparse (1-3952 with gaps).

    Returns
    -------
    ratings_df  : DataFrame with new columns user_idx, movie_idx
    num_users   : total unique users
    num_movies  : total unique movies
    """
    # Collect unique IDs (small enough to fit in driver memory)
    user_ids  = sorted([r["user_id"]  for r in ratings_df.select("user_id").distinct().collect()])
    movie_ids = sorted([r["movie_id"] for r in ratings_df.select("movie_id").distinct().collect()])

    user_map  = {uid: idx for idx, uid in enumerate(user_ids)}
    movie_map = {mid: idx for idx, mid in enumerate(movie_ids)}

    num_users  = len(user_map)
    num_movies = len(movie_map)

    print(f"[Data] Unique users: {num_users:,}  |  Unique movies: {num_movies:,}")

    # Broadcast the maps so every Spark worker can apply them without shuffling
    spark = SparkSession.getActiveSession()
    bc_user_map  = spark.sparkContext.broadcast(user_map)
    bc_movie_map = spark.sparkContext.broadcast(movie_map)

    # UDFs to remap IDs
    map_user  = F.udf(lambda uid: bc_user_map.value[uid],  IntegerType())
    map_movie = F.udf(lambda mid: bc_movie_map.value[mid], IntegerType())

    ratings_df = (
        ratings_df
        .withColumn("user_idx",  map_user(F.col("user_id")))
        .withColumn("movie_idx", map_movie(F.col("movie_id")))
    )

    return ratings_df, num_users, num_movies, user_map, movie_map


# ---------------------------------------------------------------------------
# 1D. Distributed 80/20 Train-Test Split (PySpark native)
# ---------------------------------------------------------------------------

def split_data(ratings_df, train_ratio: float = 0.8, seed: int = 42):
    """
    Use PySpark's built-in randomSplit — the split is distributed,
    meaning each partition is split independently in parallel.

    Parameters
    ----------
    ratings_df  : full ratings DataFrame
    train_ratio : fraction for training  (default 0.8)
    seed        : random seed for reproducibility

    Returns
    -------
    (train_df, test_df)  — both as PySpark DataFrames
    """
    test_ratio = 1.0 - train_ratio
    train_df, test_df = ratings_df.randomSplit(
        weights=[train_ratio, test_ratio],
        seed=seed
    )

    # Cache both partitions — we will read them multiple times
    train_df = train_df.cache()
    test_df  = test_df.cache()

    train_count = train_df.count()
    test_count  = test_df.count()

    print(f"[Split] Train: {train_count:,} rows  |  Test: {test_count:,} rows  "
          f"|  Ratio ≈ {train_count/(train_count+test_count):.2%} / "
          f"{test_count/(train_count+test_count):.2%}")

    return train_df, test_df


# ---------------------------------------------------------------------------
# 1E. PyTorch Dataset — bridge between PySpark DF and DataLoader
# ---------------------------------------------------------------------------

class RatingsDataset(Dataset):
    """
    Converts a collected PySpark DataFrame (list of Row objects) into
    a PyTorch Dataset so it can be fed into a standard DataLoader.

    For very large datasets the recommended pattern is to write the
    PySpark DataFrame to Parquet partitions and stream them; for the
    1M-row MovieLens dataset collecting to the driver is efficient.
    """

    def __init__(self, spark_df):
        """
        Parameters
        ----------
        spark_df : PySpark DataFrame with columns [user_idx, movie_idx, rating]
        """
        # Collect only the three columns we need to minimise memory
        rows = spark_df.select("user_idx", "movie_idx", "rating").collect()

        self.user_idx  = torch.tensor([r["user_idx"]  for r in rows], dtype=torch.long)
        self.movie_idx = torch.tensor([r["movie_idx"] for r in rows], dtype=torch.long)
        self.ratings   = torch.tensor([r["rating"]    for r in rows], dtype=torch.float32)

        print(f"[Dataset] Materialised {len(self.ratings):,} samples into tensors")

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_idx[idx], self.movie_idx[idx], self.ratings[idx]


def create_dataloaders(
    train_df,
    test_df,
    batch_size: int = 1024,
    num_workers: int = 4,
):
    """
    Convert PySpark DataFrames → PyTorch Datasets → DataLoaders.

    Parameters
    ----------
    train_df    : PySpark training DataFrame
    test_df     : PySpark test DataFrame
    batch_size  : mini-batch size (1024 is a good default for 1M rows)
    num_workers : parallel workers for DataLoader prefetching

    Returns
    -------
    (train_loader, test_loader)
    """
    train_dataset = RatingsDataset(train_df)
    test_dataset  = RatingsDataset(test_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,               # shuffle each epoch for better SGD convergence
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),   # speeds up CPU→GPU transfer
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,  # no gradients needed → larger batches are fine
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    print(f"[DataLoader] Train batches: {len(train_loader):,}  "
          f"|  Test batches: {len(test_loader):,}  "
          f"|  Batch size: {batch_size}")

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# 1F. Convenience wrapper — run the full pipeline in one call
# ---------------------------------------------------------------------------

def build_pipeline(data_path: str, batch_size: int = 1024):
    """
    End-to-end data pipeline:
      ratings.dat → Spark DF → clean → ID-remap → split → DataLoaders

    Returns
    -------
    dict with keys:
        spark, train_loader, test_loader,
        num_users, num_movies,
        train_df, test_df  (PySpark — kept for Spark-native evaluation)
    """
    spark       = create_spark_session()
    ratings_df  = load_ratings(spark, data_path)
    ratings_df, num_users, num_movies, user_map, movie_map = build_id_maps(ratings_df)
    train_df, test_df = split_data(ratings_df)
    train_loader, test_loader = create_dataloaders(
        train_df, test_df, batch_size=batch_size
    )

    return {
        "spark":        spark,
        "train_loader": train_loader,
        "test_loader":  test_loader,
        "num_users":    num_users,
        "num_movies":   num_movies,
        "train_df":     train_df,
        "test_df":      test_df,
        "user_map":     user_map,
        "movie_map":    movie_map,
    }

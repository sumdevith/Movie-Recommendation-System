"""
=============================================================================
SECTION 1: Distributed Data Pipeline (PySpark)
=============================================================================
MovieLens 1M Dataset — Demographic + Genre Recommender
Pipeline: Load ratings/users/movies → Join → Engineer features → Split →
          Convert to PyTorch DataLoaders

Files used (all '::'-separated):
    ratings.dat  UserID::MovieID::Rating::Timestamp     (1,000,209 rows)
    users.dat    UserID::Gender::Age::Occupation::Zip   (6,040 rows)
    movies.dat   MovieID::Title::Genres                 (3,883 rows)

Each training sample is a *content* feature vector — no user/movie IDs:
    (gender_idx, age_idx, occ_idx, zip_idx, genre_multihot[18])  →  rating
=============================================================================
"""

import multiprocessing
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, IntegerType, FloatType, LongType, StringType,
    ArrayType,
)

import features as feat


# ---------------------------------------------------------------------------
# 1A. SparkSession — maximise all available cores, tune memory
# ---------------------------------------------------------------------------

def create_spark_session(app_name: str = "DemographicRecommender") -> SparkSession:
    """Initialise a local SparkSession that uses every available CPU core."""
    num_cores = multiprocessing.cpu_count()
    master    = f"local[{num_cores}]"

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.driver.memory",          "4g")
        .config("spark.executor.memory",        "4g")
        .config("spark.driver.maxResultSize",   "2g")
        .config("spark.sql.shuffle.partitions", str(num_cores * 2))
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    print(f"[Spark] Session started  |  master={master}  |  cores={num_cores}")
    print(f"[Spark] Spark version: {spark.version}")
    return spark


# ---------------------------------------------------------------------------
# 1B. Load the three '::'-separated MovieLens files
# ---------------------------------------------------------------------------

def _read_dat(spark, path, columns, casts):
    """Read a '::'-separated .dat file into a typed DataFrame.

    Spark's CSV reader does not support multi-character separators, so we read
    raw text and split on '::' manually — reliable across all Spark versions.
    """
    raw = spark.read.text(str(path))
    parts = raw.select(F.split(F.col("value"), "::").alias("p"))
    return parts.select(
        *[F.col("p")[i].cast(cast).alias(name)
          for i, (name, cast) in enumerate(zip(columns, casts))]
    )


def load_frames(spark: SparkSession, data_dir: str):
    """Load ratings, users and movies DataFrames."""
    data_dir = Path(data_dir)

    ratings = _read_dat(
        spark, data_dir / "ratings.dat",
        ["user_id", "movie_id", "rating", "timestamp"],
        [IntegerType(), IntegerType(), FloatType(), LongType()],
    ).dropna(subset=["user_id", "movie_id", "rating"])

    users = _read_dat(
        spark, data_dir / "users.dat",
        ["user_id", "gender", "age", "occupation", "zip"],
        [IntegerType(), StringType(), IntegerType(), IntegerType(), StringType()],
    ).dropna(subset=["user_id"])

    movies = _read_dat(
        spark, data_dir / "movies.dat",
        ["movie_id", "title", "genres"],
        [IntegerType(), StringType(), StringType()],
    ).dropna(subset=["movie_id"])

    print(f"[Data] ratings={ratings.count():,}  users={users.count():,}  movies={movies.count():,}")
    return ratings, users, movies


# ---------------------------------------------------------------------------
# 1C. Feature engineering — join + encode categoricals + multi-hot genres
# ---------------------------------------------------------------------------

def _register_udfs():
    """Build Spark UDFs from the shared ``features`` helpers (single source of truth)."""
    gender_udf = F.udf(lambda g: feat.GENDER_MAP.get(g, 0), IntegerType())
    age_udf    = F.udf(lambda a: feat.AGE_MAP.get(int(a), 0) if a is not None else 0, IntegerType())
    zip_udf    = F.udf(lambda z: feat.zip_to_region(z), IntegerType())
    genre_udf  = F.udf(lambda s: feat.genres_to_multihot(s), ArrayType(FloatType()))
    return gender_udf, age_udf, zip_udf, genre_udf


def build_features(ratings, users, movies):
    """Join the three frames and produce the model-ready feature columns.

    Returns a DataFrame with:
        gender_idx, age_idx, occ_idx, zip_idx  (IntegerType)
        genres  (ArrayType(FloatType), length 18)
        rating  (FloatType)
    """
    gender_udf, age_udf, zip_udf, genre_udf = _register_udfs()

    df = (
        ratings
        .join(users,  on="user_id",  how="inner")
        .join(movies, on="movie_id", how="inner")
        .select(
            gender_udf(F.col("gender")).alias("gender_idx"),
            age_udf(F.col("age")).alias("age_idx"),
            F.col("occupation").alias("occ_idx"),
            zip_udf(F.col("zip")).alias("zip_idx"),
            genre_udf(F.col("genres")).alias("genres"),
            F.col("rating"),
        )
    )

    print("[Data] Engineered feature schema:")
    df.printSchema()
    df.show(5, truncate=False)
    return df


# ---------------------------------------------------------------------------
# 1D. Distributed 80/20 Train-Test Split
# ---------------------------------------------------------------------------

def split_data(df, train_ratio: float = 0.8, seed: int = 42):
    """PySpark native randomSplit — each partition is split in parallel."""
    train_df, test_df = df.randomSplit([train_ratio, 1.0 - train_ratio], seed=seed)
    train_df = train_df.cache()
    test_df  = test_df.cache()

    train_count, test_count = train_df.count(), test_df.count()
    total = train_count + test_count
    print(f"[Split] Train: {train_count:,}  |  Test: {test_count:,}  "
          f"|  Ratio ≈ {train_count/total:.2%} / {test_count/total:.2%}")
    return train_df, test_df


# ---------------------------------------------------------------------------
# 1E. PyTorch Dataset — bridge between PySpark DF and DataLoader
# ---------------------------------------------------------------------------

class RatingsDataset(Dataset):
    """Materialise a feature DataFrame into PyTorch tensors.

    Each item is a 6-tuple:
        (gender, age, occupation, zip_region, genres[18], rating)
    """

    def __init__(self, spark_df):
        rows = spark_df.select(
            "gender_idx", "age_idx", "occ_idx", "zip_idx", "genres", "rating"
        ).collect()

        self.gender  = torch.tensor([r["gender_idx"] for r in rows], dtype=torch.long)
        self.age     = torch.tensor([r["age_idx"]    for r in rows], dtype=torch.long)
        self.occ     = torch.tensor([r["occ_idx"]    for r in rows], dtype=torch.long)
        self.zip     = torch.tensor([r["zip_idx"]    for r in rows], dtype=torch.long)
        self.genres  = torch.tensor([list(r["genres"]) for r in rows], dtype=torch.float32)
        self.ratings = torch.tensor([r["rating"]     for r in rows], dtype=torch.float32)

        print(f"[Dataset] Materialised {len(self.ratings):,} samples into tensors")

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (
            self.gender[idx],
            self.age[idx],
            self.occ[idx],
            self.zip[idx],
            self.genres[idx],
            self.ratings[idx],
        )


def create_dataloaders(train_df, test_df, batch_size: int = 1024, num_workers: int = 4):
    """Convert PySpark DataFrames → PyTorch Datasets → DataLoaders."""
    train_dataset = RatingsDataset(train_df)
    test_dataset  = RatingsDataset(test_df)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=False,
    )

    print(f"[DataLoader] Train batches: {len(train_loader):,}  "
          f"|  Test batches: {len(test_loader):,}  |  Batch size: {batch_size}")
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# 1F. Convenience wrapper — run the full pipeline in one call
# ---------------------------------------------------------------------------

def build_pipeline(data_dir: str = "data", batch_size: int = 1024):
    """End-to-end data pipeline:
        ratings/users/movies → Spark join → features → split → DataLoaders

    Returns
    -------
    dict with keys:
        spark, train_loader, test_loader,
        feature_dims,                 (cardinalities for build_model)
        train_df, test_df             (PySpark — kept for Spark-native evaluation)
    """
    spark = create_spark_session()
    ratings, users, movies = load_frames(spark, data_dir)
    df = build_features(ratings, users, movies)
    train_df, test_df = split_data(df)
    train_loader, test_loader = create_dataloaders(train_df, test_df, batch_size=batch_size)

    return {
        "spark":         spark,
        "train_loader":  train_loader,
        "test_loader":   test_loader,
        "feature_dims":  feat.feature_dims(),
        "train_df":      train_df,
        "test_df":       test_df,
    }

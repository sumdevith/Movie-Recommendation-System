"""
=============================================================================
SHARED FEATURE DEFINITIONS — Demographic + Genre Recommender
=============================================================================
Single source of truth for how MovieLens 1M raw fields are turned into the
categorical / multi-hot features consumed by the model.

The model no longer uses user IDs or movie IDs. Instead it predicts a rating
from:

    USER side  : Gender, Age group, Occupation, Zip-code region
    MOVIE side : Genres (multi-hot over the 18 MovieLens genres)

Every component (Spark data pipeline, distributed training, Spark evaluation,
and the FastAPI server) imports these helpers so the feature encoding is
guaranteed to be identical at train time and inference time.
=============================================================================
"""

# ---------------------------------------------------------------------------
# Gender  —  "M"/"F"  →  index
# ---------------------------------------------------------------------------
GENDER_MAP = {"M": 0, "F": 1}
NUM_GENDER = len(GENDER_MAP)                       # 2

# Human-readable labels (used by the UI / slides)
GENDER_LABELS = {0: "Male", 1: "Female"}


# ---------------------------------------------------------------------------
# Age  —  MovieLens uses 7 discrete age-range codes  →  contiguous index
# ---------------------------------------------------------------------------
AGE_MAP = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}
NUM_AGE = len(AGE_MAP)                             # 7

AGE_LABELS = {
    0: "Under 18",
    1: "18-24",
    2: "25-34",
    3: "35-44",
    4: "45-49",
    5: "50-55",
    6: "56+",
}


# ---------------------------------------------------------------------------
# Occupation  —  already a contiguous 0-20 code in users.dat
# ---------------------------------------------------------------------------
OCCUPATION_LABELS = {
    0: "other / not specified",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer",
}
NUM_OCCUPATION = len(OCCUPATION_LABELS)            # 21


# ---------------------------------------------------------------------------
# Zip-code  —  collapsed to its first digit (US postal region 0-9).
#              Non-numeric / missing zips map to a dedicated "unknown" bucket.
# ---------------------------------------------------------------------------
ZIP_UNKNOWN = 10
NUM_ZIP = 11                                       # regions 0-9 + unknown


def zip_to_region(zip_code) -> int:
    """Map a raw zip-code string to a 0-9 region index (10 = unknown).

    MovieLens zips can carry a 4-digit extension ("12345-6789") or, rarely,
    be non-numeric. We only keep the leading digit, which encodes the broad
    US geographic region and keeps the feature low-cardinality.
    """
    if zip_code is None:
        return ZIP_UNKNOWN
    text = str(zip_code).strip()
    if text and text[0].isdigit():
        return int(text[0])
    return ZIP_UNKNOWN


# ---------------------------------------------------------------------------
# Genres  —  18 MovieLens genres  →  multi-hot vector
# ---------------------------------------------------------------------------
GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]
NUM_GENRES = len(GENRES)                           # 18
GENRE_TO_IDX = {genre: idx for idx, genre in enumerate(GENRES)}


def genres_to_multihot(genre_string) -> list:
    """Convert a pipe-separated genre string to an 18-dim multi-hot list.

    Example: "Action|Sci-Fi" -> [1,0,...,1,...] (floats).
    Unknown genres are ignored.
    """
    vector = [0.0] * NUM_GENRES
    if not genre_string:
        return vector
    for genre in str(genre_string).split("|"):
        idx = GENRE_TO_IDX.get(genre.strip())
        if idx is not None:
            vector[idx] = 1.0
    return vector


def selected_genres_to_multihot(genres) -> list:
    """Multi-hot from an explicit list of genre names (the user's 1-3 picks)."""
    vector = [0.0] * NUM_GENRES
    for genre in genres:
        idx = GENRE_TO_IDX.get(str(genre).strip())
        if idx is not None:
            vector[idx] = 1.0
    return vector


# ---------------------------------------------------------------------------
# Convenience — the cardinalities the model needs, in one dict
# ---------------------------------------------------------------------------
def feature_dims() -> dict:
    """Return the categorical cardinalities consumed by ``build_model``."""
    return {
        "num_gender": NUM_GENDER,
        "num_age": NUM_AGE,
        "num_occupation": NUM_OCCUPATION,
        "num_zip": NUM_ZIP,
        "num_genres": NUM_GENRES,
    }

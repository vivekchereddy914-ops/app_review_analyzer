import pandas as pd
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 0  # makes langdetect give the same result every run

COLUMNS_TO_DROP = ["userName", "userImage"]


# 1. Drop columns
def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=COLUMNS_TO_DROP, errors="ignore")


# 2. Remove duplicate review IDs
def remove_duplicate_ids(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset="reviewId", keep="first")


# 3. Normalise whitespace
def normalise_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["content"] = (
        df["content"]
        .astype("string")
        .str.replace(r"\s+", " ", regex=True)  # newlines, tabs, repeated spaces -> one space
        .str.strip()
    )
    # drop reviews that are now empty
    return df[df["content"].notna() & (df["content"] != "")]


# 4. Keep English only
def is_english(text: str) -> bool:
    # langdetect is unreliable on very short text ("Good", "nice app" come back as
    # Dutch, Italian...). Keep short reviews here; filter them out by length later.
    if len(text.split()) < 4:
        return True
    try:
        return detect(text) == "en"
    except LangDetectException:  # e.g. emoji-only or numbers-only text
        return False


def keep_english(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["content"].apply(is_english)]


# 5. Convert dates
def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["at"] = pd.to_datetime(df["at"], errors="coerce")
    if "repliedAt" in df.columns:
        df["repliedAt"] = pd.to_datetime(df["repliedAt"], errors="coerce")
    return df


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    steps = [drop_columns, remove_duplicate_ids, normalise_whitespace, keep_english, convert_dates]
    print(f"Raw reviews: {len(df)}")
    for step in steps:
        df = step(df)
        print(f"After {step.__name__}: {len(df)}")
    return df.reset_index(drop=True)


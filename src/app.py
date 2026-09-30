import argparse
from pathlib import Path

import pandas as pd

from cleanData import clean_reviews
from getReviews import get_reviews
from zero_shot_nli_classifier import classify_reviews

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

if __name__ == "__main__":

    # remove comment section before commit

    # First input the app id of the app you want to collect reviews for
    # PLAY_STORE_APP_ID = "com.google.android.apps.maps"
    # MAX_REVIEWS = 1000
    # print("Collecting reviews")
    # reviews = get_reviews(PLAY_STORE_APP_ID, MAX_REVIEWS)
    # # print(reviews)
    # df= pd.DataFrame(reviews)
    # # print(df)

    # RESULTS_DIR.mkdir(exist_ok=True)
    # output_path = RESULTS_DIR / "raw_reviews_data.xlsx"
    # df.to_excel(output_path, index=False)
    # print(f"Saved reviews to {output_path}")

    # Comment section before commit
    output_path = RESULTS_DIR / "raw_reviews_data.xlsx"
    df = pd.read_excel(output_path)

    cleaned = clean_reviews(df)
    # print(cleaned)

    cleaned_output_path = RESULTS_DIR / "cleaned_reviews_data.xlsx"
    cleaned.to_excel(cleaned_output_path, index=False)

    # Classify reviews
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="classify only the first N reviews")
    args = parser.parse_args()

    LABELS = [
        "ADHD, focus or feeling overwhelmed",
        "accessibility or readability problems",
        "unclear pricing, privacy or unexplained changes",
        "bugs or crashes",
        "a feature request",
        "praise",
    ]

    if args.limit:
        cleaned = cleaned.head(args.limit)

    classified = classify_reviews(cleaned, LABELS)

    print("\nReviews per label")
    print(classified["predicted_label"].value_counts())

    classified_output_path = RESULTS_DIR / "classified_reviews_data.xlsx"
    classified.to_excel(classified_output_path, index=False)
    print(f"Saved classified reviews to {classified_output_path}")

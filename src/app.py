import argparse
import re
from pathlib import Path

import pandas as pd

from cleanData import clean_reviews
from getReviews import get_reviews
from llm_verifier import verify_classifications
from zero_shot_nli_classifier import classify_reviews

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="classify and verify only the first N reviews")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    # 1. Collect reviews
    # First input the app id of the app you want to collect reviews for
    PLAY_STORE_APP_ID = "com.google.android.apps.maps"
    MAX_REVIEWS = 1000
    print("Collecting reviews")
    reviews = get_reviews(PLAY_STORE_APP_ID, MAX_REVIEWS)
    df = pd.DataFrame(reviews)

    output_path = RESULTS_DIR / "raw_reviews_data.xlsx"
    df.to_excel(output_path, index=False)
    print(f"Saved reviews to {output_path}")

    # To skip scraping while testing, comment out step 1 and load the saved file:
    # df = pd.read_excel(RESULTS_DIR / "raw_reviews_data.xlsx")

    # 2. Clean reviews
    cleaned = clean_reviews(df)

    cleaned_output_path = RESULTS_DIR / "cleaned_reviews_data.xlsx"
    cleaned.to_excel(cleaned_output_path, index=False)
    print(f"Saved cleaned reviews to {cleaned_output_path}")

    # 3. Classify reviews
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

    # Classifying all reviews takes ~25 minutes on CPU. To test only the
    # verification, comment out step 3 (except LABELS) and load the saved file:
    # classified = pd.read_excel(RESULTS_DIR / "classified_reviews_data.xlsx")

    # 4. Verify the classifications with Claude, saving after every label
    verified_output_path = RESULTS_DIR / "verified_reviews_data.xlsx"
    sheets = {}
    comments = []
    for label, verified, summary in verify_classifications(classified, LABELS):
        n_changed = int(verified["changed"].fillna(False).sum())
        print(f"  Claude changed {n_changed} of {len(verified)} reviews")
        print(f"  Claude says: {summary}")

        # Excel sheet names: max 31 characters, no []:*?/\
        sheets[re.sub(r"[\[\]:*?/\\]", "", label)[:31]] = verified
        comments.append({"label": label, "n_reviews": len(verified), "n_changed": n_changed, "llm_summary": summary})

        # rewrite the whole file so it always has every label done so far
        with pd.ExcelWriter(verified_output_path) as writer:
            for sheet_name, sheet_df in sheets.items():
                sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
            pd.DataFrame(comments).to_excel(writer, sheet_name="llm_comments", index=False)

    print(f"\nSaved verified reviews to {verified_output_path}")
    print("Run 'python src/charts.py' to create the charts and summary table")

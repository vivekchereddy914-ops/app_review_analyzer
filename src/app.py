from pathlib import Path

import pandas as pd

from cleanData import clean_reviews
from getReviews import get_reviews

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

if __name__ == "__main__":

    # remove comment section before commit

    # First input the app id of the app you want to collect reviews for
    PLAY_STORE_APP_ID = "com.google.android.apps.maps"
    MAX_REVIEWS = 100
    print("Collecting reviews")
    reviews = get_reviews(PLAY_STORE_APP_ID, MAX_REVIEWS)
    # print(reviews)
    df= pd.DataFrame(reviews)
    # print(df)

    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / "raw_reviews_data.xlsx"
    df.to_excel(output_path, index=False)
    print(f"Saved reviews to {output_path}")

    # Comment section before commit 
    # output_path = RESULTS_DIR / "raw_reviews_data.xlsx" 
    # df = pd.read_excel(output_path)

    cleaned = clean_reviews(df)
    print(cleaned)

    cleaned_output_path = RESULTS_DIR / "cleaned_reviews_data.xlsx"
    cleaned.to_excel(cleaned_output_path, index=False)

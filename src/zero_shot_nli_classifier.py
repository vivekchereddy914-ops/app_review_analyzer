from tqdm import tqdm
from transformers import pipeline

MODEL_NAME = "facebook/bart-large-mnli"


def classify_reviews(df, labels, threshold=0.5):
    """Add predicted_label and predicted_score columns to a copy of df."""
    classifier = pipeline("zero-shot-classification", model=MODEL_NAME)

    predicted_labels = []
    predicted_scores = []
    for text in tqdm(df["content"].fillna("").astype(str), desc="Classifying reviews"):
        # empty reviews can't be classified
        if not text.strip():
            predicted_labels.append("other")
            predicted_scores.append(0.0)
            continue

        result = classifier(text, candidate_labels=labels,
                            hypothesis_template="The user is complaining about {}.",
                            multi_label=True)

        # labels come back sorted, highest score first
        best_label = result["labels"][0]
        best_score = result["scores"][0]
        predicted_labels.append(best_label if best_score >= threshold else "other")
        predicted_scores.append(best_score)

    df = df.copy()
    df["predicted_label"] = predicted_labels
    df["predicted_score"] = predicted_scores
    return df

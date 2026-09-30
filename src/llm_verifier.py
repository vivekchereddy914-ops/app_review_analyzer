import json

import anthropic
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # loads ANTHROPIC_API_KEY from the .env file in the project root

MODEL = "claude-opus-5-5"

# Only these columns are sent to Claude
COLUMNS_TO_SEND = ["reviewId", "content", "predicted_label", "predicted_score"]


def build_schema(labels):
    """The JSON shape Claude must reply with: a summary plus one row per review."""
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "reviews": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "reviewId": {"type": "string"},
                        "verified_label": {"type": "string", "enum": labels + ["other"]},
                        "verified_score": {"type": "number"},
                        "changed": {"type": "boolean"},
                        "change_reason": {"type": "string"},
                    },
                    "required": ["reviewId", "verified_label", "verified_score", "changed", "change_reason"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["summary", "reviews"],
        "additionalProperties": False,
    }


def verify_label(client, label, label_df, labels):
    """Send all reviews of one label to Claude. Returns (verified DataFrame, Claude's summary)."""
    label_list = "\n".join(f"- {l}" for l in labels + ["other"])
    reviews_json = label_df[COLUMNS_TO_SEND].to_json(orient="records", force_ascii=False)

    prompt = f"""A zero-shot classifier sorted app store reviews into these labels:
{label_list}

predicted_score is the classifier's confidence (0 to 1). Reviews whose best score was below 0.5 were put in "other".

Below are all {len(label_df)} reviews the classifier put in "{label}".
For every review, check whether the label and the score are right:
- If both are fine, keep the same label and score, set changed to false and leave change_reason empty.
- If not, give the correct label and a better score, set changed to true and explain the change in one short sentence.

Return every review, using the same reviewId.
In summary, write a few sentences on how well this label was classified and any common mistakes.

Reviews:
{reviews_json}"""

    # Streaming, because a large label can produce a long reply
    with client.messages.stream(
        model=MODEL,
        max_tokens=64000,
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": {"type": "json_schema", "schema": build_schema(labels)}},
    ) as stream:
        response = stream.get_final_message()

    if response.stop_reason != "end_turn":
        raise ValueError(f"Claude stopped early ({response.stop_reason})")

    text = next(block.text for block in response.content if block.type == "text")
    result = json.loads(text)

    changes = pd.DataFrame(result["reviews"]).drop_duplicates("reviewId")
    # match rows by reviewId, not by position
    verified = label_df.merge(changes, on="reviewId", how="left")
    return verified, result["summary"]


def verify_classifications(df, labels):
    """Verify each label with one Claude call.

    Yields (label, verified DataFrame, Claude's summary) as each label finishes,
    so the caller can save results straight away.
    """
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY loaded from .env

    for label in labels + ["other"]:
        label_df = df[df["predicted_label"] == label]
        if label_df.empty:
            continue

        print(f"\nVerifying '{label}' ({len(label_df)} reviews)...")
        try:
            verified, summary = verify_label(client, label, label_df, labels)
        except (anthropic.APIError, ValueError) as e:
            print(f"  Failed, skipping this label: {e}")
            continue

        yield label, verified, summary

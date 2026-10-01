import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save files only, no window
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "results" / "verified_reviews_data.xlsx"
COMMENTS_SHEET = "llm_comments"  # every other sheet holds one label's reviews
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
SUMMARY_PATH = PROJECT_ROOT / "results" / "summary_table.csv"

BAR_COLOR = "#2a6fdb"
LINE_COLOR = "#d62728"
TEXT_COLOR = "#333333"


def load_data():
    """Combine the per-label sheets into one DataFrame."""
    sheets = pd.read_excel(INPUT_PATH, sheet_name=None)
    sheets.pop(COMMENTS_SHEET)
    # drop columns that are empty in a sheet (e.g. no replies) to keep concat quiet
    df = pd.concat([sheet.dropna(axis=1, how="all") for sheet in sheets.values()], ignore_index=True)
    # Claude's label is the final label; fall back to the classifier's if missing
    df["final_label"] = df["verified_label"].fillna(df["predicted_label"])
    # Claude agreed with the classifier if it kept the same label
    df["llm_agrees"] = (df["verified_label"] == df["predicted_label"]).astype(float)
    df.loc[df["verified_label"].isna(), "llm_agrees"] = float("nan")  # no verdict: leave out
    return df


def build_summary(df):
    """One row per final label, largest category first."""
    summary = df.groupby("final_label").agg(
        count=("final_label", "size"),
        avg_rating=("score", "mean"),
        agreement=("llm_agrees", "mean"),
    )
    summary["percent"] = 100 * summary["count"] / summary["count"].sum()
    summary["agreement"] = 100 * summary["agreement"]
    return summary.sort_values("count", ascending=False)


def wrap(label):
    """Labels longer than 30 characters go onto two lines."""
    return textwrap.fill(label, 30) if len(label) > 30 else label


def new_chart(summary):
    """Horizontal bar chart setup shared by all charts (largest category at the top)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels([wrap(label) for label in summary.index])
    ax.invert_yaxis()
    return fig, ax


def save(fig, filename):
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")  # keeps bar-end labels inside the image
    plt.close(fig)


def chart_reviews_per_category(summary):
    fig, ax = new_chart(summary)
    ax.barh(range(len(summary)), summary["count"], color=BAR_COLOR)
    for i, (count, percent) in enumerate(zip(summary["count"], summary["percent"])):
        ax.text(count, i, f" {count} ({percent:.0f}%)", va="center", color=TEXT_COLOR)
    ax.set_xlim(0, summary["count"].max() * 1.2)  # room for the labels
    ax.set_xlabel("Number of reviews")
    ax.set_title(f"User concerns by category (n = {summary['count'].sum()})")
    save(fig, "reviews_per_category.png")


def chart_rating_per_category(summary, overall_avg):
    fig, ax = new_chart(summary)
    # bars start at 1 because the rating scale starts at 1
    ax.barh(range(len(summary)), summary["avg_rating"] - 1, left=1, color=BAR_COLOR)
    for i, (rating, count) in enumerate(zip(summary["avg_rating"], summary["count"])):
        # white box so the average line doesn't run through the label
        ax.annotate(f"{rating:.1f} (n={count})", (rating, i), xytext=(5, 0), textcoords="offset points",
                    va="center", color=TEXT_COLOR,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1), zorder=3)
    ax.axvline(overall_avg, color=LINE_COLOR, linestyle="--", label=f"Overall avg: {overall_avg:.2f}")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(1, 5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Average star rating")
    ax.set_title("Average rating by concern")
    save(fig, "rating_per_category.png")


def chart_llm_agreement(summary):
    fig, ax = new_chart(summary)
    ax.barh(range(len(summary)), summary["agreement"], color=BAR_COLOR)
    for i, agreement in enumerate(summary["agreement"]):
        ax.text(agreement, i, f" {agreement:.0f}%", va="center", color=TEXT_COLOR)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
    ax.set_xlabel("Reviews where the LLM kept the classifier's label")
    ax.set_title("LLM agreement with NLI classifier")
    save(fig, "llm_agreement.png")


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    summary = build_summary(df)

    chart_reviews_per_category(summary)
    chart_rating_per_category(summary, df["score"].mean())
    chart_llm_agreement(summary)

    table = summary.reset_index().rename(columns={
        "final_label": "category", "percent": "%", "avg_rating": "avg rating", "agreement": "agreement %",
    })[["category", "count", "%", "avg rating", "agreement %"]].round(1)
    print(table.to_string(index=False))
    table.to_csv(SUMMARY_PATH, index=False)
    print(f"\nSaved charts to {FIGURES_DIR}")
    print(f"Saved summary table to {SUMMARY_PATH}")

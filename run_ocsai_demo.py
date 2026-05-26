"""OCSAI-lite demo for scoring Alternative Uses Task responses from a CSV file.

The official OCSAI package includes scorer classes that call fine-tuned OpenAI
models. Those fine-tuned model weights are not included in this repository, so
you need OpenAI account access to the listed model IDs to run official OCSAI.

This script provides an offline-friendly "OCSAI-lite" evaluator:

1. Read AUT responses from aut_responses.csv.
2. Convert each response and question to sentence embeddings.
3. Calculate novelty within each target_object group only.
4. Calculate usefulness against the question for that response.
5. Average novelty and usefulness into a creativity score.
6. Save the scored table to ocsai_lite_results.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ocsai.inference import Chat_Scorer, Classic_Scorer  # noqa: F401
from ocsai.inference.chat_scorer import GPTCHATMODELS
from ocsai.inference.classic_scorer import GPTCLASSICMODELS


INPUT_CSV = Path("aut_responses.csv")
OUTPUT_CSV = Path("ocsai_lite_results.csv")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

REQUIRED_COLUMNS = ["participant_id", "target_object", "question", "response"]
SCORE_COLUMNS = ["novelty_score", "usefulness_score", "creativity_score"]


def print_available_ocsai_tools() -> None:
    """Print official OCSAI classes and model IDs for reference."""
    print("Available official OCSAI scorer classes:")
    print("- Chat_Scorer: chat-completions based scorer")
    print("- Classic_Scorer: legacy completions based scorer")
    print()

    print("Bundled chat model aliases:")
    for alias, model_id in GPTCHATMODELS.items():
        print(f"- {alias}: {model_id}")
    print()

    print("Bundled classic model aliases:")
    for alias, model_id in GPTCLASSICMODELS.items():
        print(f"- {alias}: {model_id}")
    print()

    print("Official model weights are not included in this repository.")
    print("Running OCSAI-lite with sentence-transformer embeddings instead.")
    print()


def load_aut_responses(input_csv: Path) -> pd.DataFrame:
    """Load AUT responses and check that the CSV has the expected columns."""
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Could not find {input_csv}. Create it with columns: "
            f"{', '.join(REQUIRED_COLUMNS)}"
        )

    df = pd.read_csv(input_csv)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "Input CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    # Keep only the columns used by this demo, and remove rows with blank data.
    df = df[REQUIRED_COLUMNS].copy()
    for col in REQUIRED_COLUMNS:
        df[col] = df[col].astype(str).str.strip()

    df = df.replace("", np.nan).dropna(subset=REQUIRED_COLUMNS)
    return df.reset_index(drop=True)


def scale_0_to_100(values: np.ndarray) -> np.ndarray:
    """Convert 0-1 style values into readable 0-100 scores.

    Cosine similarity can sometimes be negative. For this beginner-friendly
    demo, negative values are clipped to 0 before converting to 0-100.
    """
    clipped = np.clip(values.astype(float), 0.0, 1.0)
    return np.round(clipped * 100, 2)


def score_target_group(
    group: pd.DataFrame,
    model: SentenceTransformer,
) -> pd.DataFrame:
    """Score one target_object group, such as only brick responses.

    novelty_score is calculated by comparing each response with the other
    responses in the same target_object group. This means brick responses are
    compared only with other brick responses, paperclip only with paperclip,
    and fork only with fork.
    """
    group = group.copy().reset_index(drop=True)
    responses = group["response"].tolist()
    questions = group["question"].tolist()

    # Convert text into embedding vectors. A vector is a list of numbers that
    # places semantically similar sentences near each other.
    response_embeddings = model.encode(responses)
    question_embeddings = model.encode(questions)

    # This matrix compares every response in the group with every other
    # response in that same group.
    response_similarity = cosine_similarity(response_embeddings)

    novelty_values = []
    for row_index in range(len(group)):
        if len(group) == 1:
            # With one response in a group, there are no peer responses to
            # compare against. We assign maximum novelty as a simple convention.
            novelty_values.append(1.0)
            continue

        # Drop the diagonal self-comparison, because every response is perfectly
        # similar to itself and that would distort novelty.
        other_similarities = np.delete(response_similarity[row_index], row_index)
        mean_similarity_to_others = np.mean(other_similarities)
        novelty_values.append(1.0 - mean_similarity_to_others)

    novelty_values = np.array(novelty_values)

    # Usefulness compares each response to its own question. If all rows for a
    # target share the same question, this still works exactly as expected.
    usefulness_values = np.diag(cosine_similarity(response_embeddings, question_embeddings))

    # Creativity balances being different from peers and still relevant.
    creativity_values = (novelty_values + usefulness_values) / 2

    group["novelty_score"] = scale_0_to_100(novelty_values)
    group["usefulness_score"] = scale_0_to_100(usefulness_values)
    group["creativity_score"] = scale_0_to_100(creativity_values)
    group[SCORE_COLUMNS] = group[SCORE_COLUMNS].round(2)

    return group


def score_ocsai_lite(input_csv: Path = INPUT_CSV, output_csv: Path = OUTPUT_CSV) -> pd.DataFrame:
    """Load a CSV, score all target_object groups, and save the results."""
    df = load_aut_responses(input_csv)

    # The first run may download the model from Hugging Face. After that, it
    # usually loads from the local cache.
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    scored_groups = []
    for target_object, group in df.groupby("target_object", sort=False):
        print(f"Scoring target_object={target_object!r} with {len(group)} responses")
        scored_groups.append(score_target_group(group, model))

    results = pd.concat(scored_groups, ignore_index=True)
    results = results[
        [
            "participant_id",
            "target_object",
            "question",
            "response",
            "novelty_score",
            "usefulness_score",
            "creativity_score",
        ]
    ]

    results.to_csv(output_csv, index=False)
    return results


def main() -> None:
    print_available_ocsai_tools()

    results = score_ocsai_lite()

    print()
    print("OCSAI-lite scores:")
    print(results.to_string(index=False))
    print()
    print(f"Saved CSV: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()

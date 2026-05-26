"""Small reusable scoring module for the OCSAI-lite web app.

README-style quick start
------------------------
Install dependencies:

    python -m pip install fastapi uvicorn jinja2 python-multipart sentence-transformers scikit-learn pandas numpy

Run the web app:

    uvicorn app:app --reload

What this file does
-------------------
This is not the official fine-tuned OCSAI model. It is a lightweight research
demo that uses sentence embeddings and cosine similarity:

- novelty_score: higher when a response is less similar to other responses for
  the same target_object.
- usefulness_score: higher when a response is semantically close to the AUT
  question.
- creativity_score: the average of novelty_score and usefulness_score.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RESULT_COLUMNS = [
    "participant_id",
    "target_object",
    "question",
    "response",
    "novelty_score",
    "usefulness_score",
    "creativity_score",
]


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it for every web request.

    Loading the model can take a few seconds. The lru_cache decorator keeps the
    loaded model in memory so each submit click does not reload it from disk.
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def scale_0_to_100(value: float) -> float:
    """Convert a 0-1 cosine-style score into a clean 0-100 score."""
    clipped = float(np.clip(value, 0.0, 1.0))
    return round(clipped * 100, 2)


def score_single_response(
    target_object: str,
    question: str,
    response: str,
    previous_rows: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Score one new AUT response.

    previous_rows should contain earlier saved responses. Novelty is calculated
    only against rows with the same target_object, so brick responses are not
    compared with paperclip or fork responses.
    """
    model = get_embedding_model()

    if previous_rows is None or previous_rows.empty:
        same_target_responses: list[str] = []
    else:
        same_target = previous_rows[
            previous_rows["target_object"].astype(str).str.lower()
            == target_object.lower()
        ]
        same_target_responses = same_target["response"].dropna().astype(str).tolist()

    # Put the new response last. Later we will read the final row of the
    # similarity matrix to get this response's similarity to earlier responses.
    comparison_responses = same_target_responses + [response]
    response_embeddings = model.encode(comparison_responses)

    if len(comparison_responses) == 1:
        # No peer responses exist yet for this target, so we use maximum novelty
        # as a clear convention for the first submitted answer.
        novelty_raw = 1.0
    else:
        similarity_matrix = cosine_similarity(response_embeddings)
        new_response_similarities = similarity_matrix[-1, :-1]
        novelty_raw = 1.0 - float(np.mean(new_response_similarities))

    # Usefulness compares the new response directly to the question prompt.
    question_embedding = model.encode([question])
    new_response_embedding = response_embeddings[-1].reshape(1, -1)
    usefulness_raw = float(cosine_similarity(new_response_embedding, question_embedding)[0][0])

    creativity_raw = (novelty_raw + usefulness_raw) / 2

    return {
        "novelty_score": scale_0_to_100(novelty_raw),
        "usefulness_score": scale_0_to_100(usefulness_raw),
        "creativity_score": scale_0_to_100(creativity_raw),
    }


def average_creativity_by_target(results_df: pd.DataFrame) -> dict[str, float]:
    """Return average creativity for the three supported AUT target objects."""
    averages = {"brick": 0.0, "paperclip": 0.0, "fork": 0.0}
    if results_df.empty:
        return averages

    grouped = results_df.groupby("target_object")["creativity_score"].mean()
    for target in averages:
        if target in grouped:
            averages[target] = round(float(grouped[target]), 2)

    return averages

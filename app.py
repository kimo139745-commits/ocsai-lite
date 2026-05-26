"""FastAPI web app for the OCSAI-lite creativity evaluator.

README-style quick start
------------------------
Install dependencies:

    python -m pip install fastapi uvicorn jinja2 python-multipart sentence-transformers scikit-learn pandas numpy

Run locally:

    uvicorn app:app --reload

Then open:

    http://127.0.0.1:8000
"""

from __future__ import annotations

import csv
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


BASE_DIR = Path(__file__).resolve().parent
RESULTS_CSV = BASE_DIR / "web_ocsai_lite_results.csv"
RESULT_COLUMNS = [
    "participant_id",
    "target_object",
    "question",
    "response",
    "novelty_score",
    "usefulness_score",
    "creativity_score",
]

print("APP IMPORTED")

app = FastAPI(title="AI Creativity Evaluator (OCSAI-lite)")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")
print("FASTAPI APP CREATED")


def ensure_results_csv() -> None:
    """Create the results CSV with headers if it does not exist yet."""
    if not RESULTS_CSV.exists():
        with RESULTS_CSV.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=RESULT_COLUMNS)
            writer.writeheader()


def read_results() -> list[dict[str, str]]:
    """Read saved web results from disk."""
    ensure_results_csv()
    with RESULTS_CSV.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def append_result(row: dict[str, object]) -> list[dict[str, str]]:
    """Append one scored response to web_ocsai_lite_results.csv."""
    ensure_results_csv()
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_COLUMNS)
        writer.writerow({column: row.get(column, "") for column in RESULT_COLUMNS})
    return read_results()


def average_creativity_by_target(rows: list[dict[str, str]]) -> dict[str, float]:
    """Calculate average creativity without importing the ML scoring module."""
    totals = {"brick": 0.0, "paperclip": 0.0, "fork": 0.0}
    counts = {"brick": 0, "paperclip": 0, "fork": 0}

    for row in rows:
        target = str(row.get("target_object", "")).lower()
        if target not in totals:
            continue
        try:
            score = float(row.get("creativity_score", 0))
        except (TypeError, ValueError):
            continue
        totals[target] += score
        counts[target] += 1

    return {
        target: round(totals[target] / counts[target], 2) if counts[target] else 0.0
        for target in totals
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """Render the main page with the current average creativity values."""
    results = read_results()
    averages = average_creativity_by_target(results)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "averages": averages,
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Fast health check for Render.

    This endpoint does not load the sentence-transformer model. Render can call
    it immediately after startup to confirm that the web port is open.
    """
    return {"status": "ok"}


@app.post("/score")
def score(
    participant_id: str = Form(...),
    target_object: str = Form(...),
    question: str = Form(...),
    response: str = Form(...),
) -> JSONResponse:
    """Score one form submission and append it to the results CSV."""
    # Import the scoring code only when the user actually submits a response.
    # This keeps Render startup fast enough for the port detector.
    from ocsai_lite import score_single_response

    participant_id = participant_id.strip()
    target_object = target_object.strip().lower()
    question = question.strip()
    response = response.strip()

    if target_object not in {"brick", "paperclip", "fork"}:
        return JSONResponse({"error": "Target object must be brick, paperclip, or fork."}, status_code=400)

    if not participant_id or not question or not response:
        return JSONResponse({"error": "Participant ID, question, and response are required."}, status_code=400)

    previous_results = read_results()
    scores = score_single_response(
        target_object=target_object,
        question=question,
        response=response,
        previous_rows=previous_results,
    )

    saved_row = {
        "participant_id": participant_id,
        "target_object": target_object,
        "question": question,
        "response": response,
        **scores,
    }
    updated_results = append_result(saved_row)
    averages = average_creativity_by_target(updated_results)

    return JSONResponse({"scores": scores, "averages": averages, "saved_row": saved_row})

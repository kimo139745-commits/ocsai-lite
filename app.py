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

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ocsai_lite import RESULT_COLUMNS, average_creativity_by_target, score_single_response


BASE_DIR = Path(__file__).resolve().parent
RESULTS_CSV = BASE_DIR / "web_ocsai_lite_results.csv"

app = FastAPI(title="AI Creativity Evaluator (OCSAI-lite)")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def ensure_results_csv() -> None:
    """Create the results CSV with headers if it does not exist yet."""
    if not RESULTS_CSV.exists():
        pd.DataFrame(columns=RESULT_COLUMNS).to_csv(RESULTS_CSV, index=False)


def read_results() -> pd.DataFrame:
    """Read saved web results from disk."""
    ensure_results_csv()
    return pd.read_csv(RESULTS_CSV)


def append_result(row: dict[str, object]) -> pd.DataFrame:
    """Append one scored response to web_ocsai_lite_results.csv."""
    existing = read_results()
    new_row = pd.DataFrame([row], columns=RESULT_COLUMNS)
    if existing.empty:
        updated = new_row
    else:
        updated = pd.concat([existing, new_row], ignore_index=True)
    updated.to_csv(RESULTS_CSV, index=False)
    return updated


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


@app.post("/score")
def score(
    participant_id: str = Form(...),
    target_object: str = Form(...),
    question: str = Form(...),
    response: str = Form(...),
) -> JSONResponse:
    """Score one form submission and append it to the results CSV."""
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

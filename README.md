# Ocsai-Py

These are the materials for training and analyzing Ocsai - a model for evaluating various Originality Scoring tasks with Large Language models.

This includes:
- resources pertaining to the training of Ocsai 2
- tools for training automated scoring models
- [materials related to confidence measures and weighted probabilistic scoring](./notebooks/evaluation/LogProbsOcsai1.ipynb).


## Further Information
### Citations 

Forthcoming

### Data

Ocsai is trained on a great deal of data, from a number of past studies. The training data for the paper [Beyond semantic distance: Automated scoring of divergent thinking greatly improves with large language models](https://github.com/massivetexts/ocsai/tree/main/data/ocsai1) is in the data folder, but please see the Dataset preparation notebook at [cleanDatasets.ipynb](./notebooks/ocsai2-dataprep/cleanDatasets.ipynb) to see all include datasets in Ocsai2 and citations.

### Other links

- Associated research: [DOI: 10.13140/RG.2.2.32393.31840](https://doi.org/10.13140/RG.2.2.32393.31840)
- Drawing Originality Image Processing Models: [ocsai-d](https://huggingface.co/collections/massivetexts/ocsai-d)


## Usage
### Live Version

The live website is accessible through [openscoring.du.edu](https://openscoring.du.edu/) with online access to test the Ocsai.

### Installing the Library

The code used in preparation is bundled in a library. It can be installed as follows: 

```
pip install git+https://www.github.com/massivetexts/ocsai.git
```

If you want a sandboxed virtual environment to try the code in this repository's notebooks:

```bash
git clone https://www.github.com/massivetexts/ocsai && cd ocsai
uv sync
```

This will make a virtual environment available in `./.venv`

## OCSAI-lite Web App

This repository also includes a small FastAPI web app named **AI Creativity Evaluator (OCSAI-lite)**.
The app entry file is `app.py`, and the FastAPI object is named `app`.

### Run Locally

Install the web app dependencies:

```bash
python -m pip install fastapi uvicorn jinja2 python-multipart sentence-transformers scikit-learn pandas numpy
```

Start the local server:

```bash
uvicorn app:app --reload
```

Open the app in your browser:

```text
http://127.0.0.1:8000
```

Submitted results are saved to `web_ocsai_lite_results.csv`. If that file does not exist,
`app.py` creates it automatically with the correct column headers.

### Deploy to Render

This project is ready for Render deployment with:

- `requirements.txt` for Python package installation
- `render.yaml` for Render Web Service settings
- `app.py` as the FastAPI entry point

Steps:

1. Push this repository to GitHub.
2. Log in to Render: https://render.com
3. Click **New +**.
4. Choose **Blueprint** if you want Render to read `render.yaml` automatically.
5. Connect the GitHub repository.
6. Confirm the service settings.
7. Deploy.

The Render start command is:

```bash
uvicorn app:app --host 0.0.0.0 --port 10000
```

The app also includes a lightweight health check endpoint:

```text
/health
```

The sentence-transformers model is loaded lazily. This means Render can open
the web port first, and the model is loaded only when someone submits the first
score request. On Render's free plan, that first score calculation can take
about 30-60 seconds while the model downloads or warms up. Later requests reuse
the cached model and should be faster.

Important: Do not hard-code local port `8000` for deployment. This Render setup
uses port `10000` in `render.yaml`.

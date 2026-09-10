[![CI](https://github.com/vudaodev/fraud-radar/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/vudaodev/fraud-radar/actions/workflows/ci.yml)
[![Deploy](https://github.com/vudaodev/fraud-radar/actions/workflows/cd.yml/badge.svg)](https://github.com/vudaodev/fraud-radar/actions/workflows/cd.yml)

# Fraud Radar

Real-time credit-card fraud scoring: an Isolation Forest behind a FastAPI endpoint, deployed to Heroku.

Live demo: https://fraud-radar-api-6fdd5570353c.herokuapp.com/docs (Swagger UI)

## What it is

An anomaly detector for card transactions, served over HTTP. The model is a scikit-learn Isolation Forest trained on the Kaggle credit-card fraud dataset (284K transactions, about 0.17% fraud). It runs in a Docker container on Heroku and ships through a GitHub Actions pipeline.

The operating threshold is -0.1218, chosen at the knee of the precision-recall curve. On held-out data that gives roughly 74.5% recall at 8.95% precision, or about 11 alerts per fraud caught. How the exact value was picked is documented in [common/anomaly.py](common/anomaly.py).

## Architecture

The API takes the 30 raw fields (Time, Amount, V1 to V28) and hands them to a fitted scikit-learn pipeline. Every transform, hour-of-day from Time and log1p of Amount, lives inside that pipeline and is pickled with it, so training and serving run the same code and cannot drift apart. The `common/` package holds the single definition of anything shared between components: the request and response schemas, the scoring convention (higher means more anomalous), and the threshold.

## Repo structure

```
api/                FastAPI service: the /score and /health endpoints
common/             shared definitions: schemas, scoring convention, threshold
model/              training script, notebooks, and the fitted pipeline artifact
streaming/          Kafka producer and consumer (planned, not yet in the repo)
scripts/            data-sampling utilities
tests/              pytest suite
data/               sample CSVs; creditcard.csv is gitignored
.github/workflows/  CI and CD
```

## Endpoints

`POST /score` takes one transaction and returns `{"score": float, "flagged": bool}`.

`GET /health` returns `{"model_loaded": bool, "threshold": float}`.

The payload below is the fraud row from [tests/test_api.py](tests/test_api.py), so this README, the test suite, and the live endpoint all agree on it.

```bash
curl -X POST https://fraud-radar-api-6fdd5570353c.herokuapp.com/score \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 93860.0, "Amount": 188.52,
    "V1": -10.6323749061596, "V2": 7.25193622855414, "V3": -17.6810718207918,
    "V4": 8.20414440620562, "V5": -10.1665907519072, "V6": -4.51034377036334,
    "V7": -12.9816061559658, "V8": 6.78358879797499, "V9": -4.65932958355558,
    "V10": -14.9246547735487, "V11": 8.38914233451929, "V12": -16.4655039422141,
    "V13": 0.33851695978655, "V14": -14.224403603167, "V15": 0.556584472572111,
    "V16": -11.683998043525, "V17": -15.8416159780561, "V18": -5.75319975278369,
    "V19": 3.81304079276336, "V20": -0.810146481561289, "V21": 2.71535704420309,
    "V22": 0.695602689761576, "V23": -1.13812206664164, "V24": 0.459442241911828,
    "V25": 0.386337323495895, "V26": 0.522438449202614, "V27": -1.41660373652915,
    "V28": -0.488307035713995
  }'
```

```json
{"score": 0.089862, "flagged": true}
```

```bash
curl https://fraud-radar-api-6fdd5570353c.herokuapp.com/health
```

```json
{"model_loaded": true, "threshold": -0.1218}
```

## Run locally

From source:

```bash
uv sync
uv run fastapi dev api/main.py    # http://127.0.0.1:8000/docs
```

With Docker:

```bash
docker build -f api/Dockerfile -t fraud-radar .
docker run --rm -p 8000:8000 fraud-radar
```

Neither path needs the dataset. `data/creditcard.csv` is a gitignored Kaggle download used only for retraining:

```bash
uv run python -m model.train    # overwrites model/artifacts/isolation_forest.pkl
```

## CI/CD

CI runs on every push to any branch: ruff lint and format check, pytest, then a Docker build. CD runs once CI passes on `main`. It builds the image, pushes it to the Heroku container registry, releases it, and smoke tests the live `/health` endpoint until it reports the model loaded.

## Tech stack

Python 3.13, FastAPI, scikit-learn, Docker (multi-stage build), uv, pytest, GitHub Actions, Heroku.

# PathSnap: map-matching noisy GPS to a road network

Given a trip's GPS track (`.track`) and its road network (`.nodes`, `.arcs`), PathSnap predicts which directed road segment (arc) the vehicle was on at every GPS observation, and reconstructs a connected route. Reference routes (`.route`) are used only for training labels and server-side scoring.

Pipeline: **candidate generation → per-candidate classifier (LightGBM, Bayesian-tuned) → HMM/Viterbi decoding over the road graph → FastAPI backend → browser UI.**

## Repository layout

| Path | What |
|---|---|
| `pathsnap/data.py` | Loading raw files, per-trip UTM projection, reference labelling (monotonic track-to-route alignment) |
| `pathsnap/features.py` | Candidate arcs per GPS point and their features |
| `pathsnap/hmm.py` | Road graph, shortest paths, Viterbi decoder, route connection |
| `pathsnap/metrics.py` | Top-1, macro P/R/F1, log loss, confusion matrix, continuity, route F1 |
| `pathsnap/pipeline.py` | End-to-end prediction for one trip + server-side evaluation |
| `scripts/eda.py` | Dataset exploration → `results/eda/` |
| `scripts/build_features.py` | Builds cached candidate tables → `features/` (not committed, regenerate) |
| `scripts/train_models.py` | 6 models, 5-fold GroupKFold CV + student-test → `results/`, `models/` |
| `scripts/tune.py` | Optuna TPE search for LightGBM → `results/hpo_*`, `models/LightGBM_tuned.joblib` |
| `scripts/hmm_eval.py` | Decoder tuning on train (out-of-fold), classifier vs HMM on student-test → `models/config.json` |
| `scripts/make_maps.py` | Folium maps (GPS, network, reference, predicted, legend, layer control) → `results/maps/` |
| `api/app.py` | FastAPI backend (laptop 1) |
| `ui/index.html` | Evaluator UI (laptop 2) |

## Setup

Python 3.12.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place the data as `data/train/<trip>/`, `data/student_test/<trip>/` (each trip folder holds `<trip>.track/.nodes/.arcs/.route`).

## Reproduce all results

Run from the repo root, in order:

```bash
python scripts/eda.py              # EDA tables and plots
python scripts/build_features.py   # candidates + features for train and student_test
python scripts/train_models.py     # model comparison (all 6 models)
python scripts/tune.py             # Bayesian HPO on LightGBM
python scripts/hmm_eval.py         # HMM decoder tuning and evaluation, writes models/config.json
python scripts/make_maps.py student_test   # Folium maps
```

All randomness is seeded (`SEED = 42`).

## Deployment (two laptops)

**Laptop 1: API.** Copy the repo, `models/`, and the evaluation trips. Point `PATHSNAP_DATA` at the folder of evaluation trips (one subfolder per trip, including `.route` for server-side scoring):

```bash
PATHSNAP_DATA=/path/to/eval_trips uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Request logs, the pipeline summary and the evaluation metrics for every request are printed in this terminal.

Endpoints:
- `GET /trips`: available trip ids
- `POST /predict` with `{"trip_id": "00000000"}`: runs the full pipeline and returns GPS points, the predicted route (arc segments as lat/lon) and metrics. The reference route itself is never returned.

**Laptop 2: UI.** Serve the static page and open it with the API address of laptop 1:

```bash
python -m http.server 8080 -d ui
# open http://localhost:8080/?api=http://<laptop1-ip>:8000
```

Both laptops must be on the same network, and laptop 1's firewall must allow port 8000.

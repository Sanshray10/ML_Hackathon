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
| `scripts/make_maps.py` | Folium maps (GPS, network, reference, predicted, legend, layer control) → `results/maps/` (open the HTML files in a browser) |
| `api/app.py` | FastAPI backend (laptop 1) |
| `ui/index.html` | Evaluator UI (laptop 2) |

## Results

Primary metric: observation-level **Top-1 accuracy**, the share of GPS points whose predicted arc equals the reference arc.

Model comparison (`results/model_comparison.csv`): 5-fold GroupKFold CV on the 80 training trips, then the 10 student-test trips.

| Model | Family | CV macro F1 | CV Top-1 | Test Top-1 | Test macro F1 | Test log loss | Train (s) |
|---|---|---|---|---|---|---|---|
| GaussianNB | Probabilistic | 0.867 | 0.870 | 0.865 | 0.856 | 0.486 | 0.3 |
| LogReg | Linear | 0.912 | 0.883 | 0.869 | 0.905 | 0.130 | 2.3 |
| KNN | Instance-based | 0.922 | 0.881 | 0.878 | 0.922 | 0.121 | 0.4 |
| RandomForest | Bagging | 0.936 | 0.900 | 0.890 | 0.933 | 0.088 | 106.5 |
| LightGBM | Boosting | 0.935 | 0.898 | 0.893 | 0.933 | 0.094 | 2.0 |
| MLP | Neural network | 0.932 | 0.895 | 0.893 | 0.931 | 0.092 | 136.3 |

Bayesian HPO (Optuna TPE, 40 trials, objective = CV Top-1; `results/hpo_*`): LightGBM CV Top-1 0.8985 → 0.9030, student-test Top-1 0.8929 → 0.8935.

HMM decoding (β = 2, tuned on out-of-fold train predictions; `results/hmm_*`), student-test:

| | Top-1 | Route F1 | Continuity |
|---|---|---|---|
| Tuned LightGBM, classifier only | 0.894 | 0.905 | 0.831 |
| Tuned LightGBM + HMM | 0.925 | 0.961 | 0.909 |
| Reference labels | | | 0.919 |

Candidate miss rate: 0.84% train, 0.26% student-test. The Random Forest weights (413 MB) exceed GitHub's file limit and are not committed; `python scripts/train_models.py RandomForest` regenerates them.

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
python scripts/candidate_stats.py          # candidate row counts and miss rates
python scripts/lgbm_learning_curve.py      # LightGBM train/validation log loss per boosting round
python scripts/evaluate_trips.py student_test     # per-trip results of the deployed pipeline
python scripts/evaluate_trips.py evaluator_test   # final evaluation (run once, after development)
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

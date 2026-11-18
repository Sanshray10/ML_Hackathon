import json
import logging
import os
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap.pipeline import evaluate_trip, predict_trip

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA", ROOT / "data" / "student_test"))
CONFIG = json.loads((ROOT / "models" / "config.json").read_text())

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pathsnap")

model = joblib.load(ROOT / "models" / CONFIG["model"])
log.info("loaded %s (beta=%s), serving trips from %s", CONFIG["model"], CONFIG["beta"], DATA_DIR)

app = FastAPI(title="PathSnap")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


def trip_ids():
    return sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())


@app.get("/trips")
def trips():
    return {"trips": trip_ids()}


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in trip_ids():
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    trip_dir = DATA_DIR / req.trip_id
    log.info("request trip=%s", req.trip_id)
    result = predict_trip(trip_dir, model, CONFIG["beta"])
    trip = result["trip"]
    log.info("trip=%s points=%d candidates=%d route_arcs=%d pipeline=%.2fs", req.trip_id, len(trip.xy),
             result["n_candidates"], len(result["route"]), result["runtime_s"])

    metrics = evaluate_trip(trip_dir, result)
    metrics["runtime_s"] = result["runtime_s"]
    log.info("trip=%s evaluation: %s", req.trip_id,
             ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()))

    nodes = trip.node_lonlat[:, ::-1].round(6)
    return {
        "trip_id": req.trip_id,
        "gps": trip.lonlat[:, ::-1].round(6).tolist(),
        "route": [[nodes[u].tolist(), nodes[v].tolist()] for u, v in trip.arcs[result["route"]]],
        "metrics": metrics,
    }

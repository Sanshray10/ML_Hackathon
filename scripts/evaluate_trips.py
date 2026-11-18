import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.data import load_route
from pathsnap.metrics import continuity
from pathsnap.pipeline import evaluate_trip, predict_trip

# Runs the deployed pipeline (same model, beta and code as the API) on every trip of a partition.
part = sys.argv[1] if len(sys.argv) > 1 else "student_test"
config = json.loads(Path("models/config.json").read_text())
model = joblib.load(Path("models") / config["model"])
rows = []
for d in sorted(Path("data", part).iterdir()):
    res = predict_trip(d, model, config["beta"])
    arcs, route = res["trip"].arcs, res["route"]
    rows.append({"trip": d.name, **evaluate_trip(d, res), "runtime_s": res["runtime_s"],
                 "route_file_continuity": continuity(load_route(d), arcs),
                 "connected_route_continuity": continuity(route, arcs),
                 "route_breaks": int((arcs[route[:-1], 1] != arcs[route[1:], 0]).sum())})
out = pd.DataFrame(rows)
out.to_csv(Path("results") / f"pipeline_{part}.csv", index=False)
print(out[["trip", "classifier_top1", "hmm_top1", "connected_route_continuity", "route_breaks"]].to_string())
labeled = out.n_labeled
print(f"pooled: classifier {(out.classifier_top1 * labeled).sum() / labeled.sum():.4f}, "
      f"hmm {(out.hmm_top1 * labeled).sum() / labeled.sum():.4f}")

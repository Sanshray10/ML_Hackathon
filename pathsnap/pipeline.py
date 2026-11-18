import time

import numpy as np

from .data import align_route, load_route, load_trip
from .features import FEATURES, build_candidates
from .hmm import BETA, decode
from .metrics import collapse, continuity, route_prf


def predict_trip(trip_dir, model, beta=BETA):
    """Full prediction pipeline for one trip. Uses only GPS + road network, never the reference route."""
    start = time.perf_counter()
    trip = load_trip(trip_dir)
    cand = build_candidates(trip)
    cand["p"] = model.predict_proba(cand[FEATURES].to_numpy())[:, 1]
    clf = np.full(len(trip.xy), -1)
    best = cand.loc[cand.groupby("point").p.idxmax()]
    clf[best.point.to_numpy()] = best.arc.to_numpy()
    hmm, route = decode(trip, cand, beta)
    return {"trip": trip, "classifier_pred": clf, "hmm_pred": hmm, "route": route,
            "n_candidates": len(cand), "runtime_s": time.perf_counter() - start}


def evaluate_trip(trip_dir, result):
    """Server-side scoring against the reference route."""
    trip, ref = result["trip"], load_route(trip_dir)
    label = align_route(trip, ref)
    labeled = label >= 0
    out = {"n_points": len(label), "n_labeled": int(labeled.sum()),
           "reference_continuity": continuity(label[labeled], trip.arcs)}
    for name in ["classifier", "hmm"]:
        pred = result[f"{name}_pred"]
        out[f"{name}_top1"] = float((pred[labeled] == label[labeled]).mean())
        out[f"{name}_continuity"] = continuity(pred[pred >= 0], trip.arcs)
        route = result["route"] if name == "hmm" else collapse(pred[pred >= 0])
        out.update({f"{name}_{k}": v for k, v in route_prf(route, ref).items()})
    return out

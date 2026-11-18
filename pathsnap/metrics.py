import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, precision_recall_fscore_support


def point_predictions(cand, score="p"):
    best = cand.loc[cand.groupby(["trip", "point"])[score].idxmax(), ["trip", "point", "arc"]]
    return best.rename(columns={"arc": "pred"})


def top1_accuracy(points, preds):
    """Share of labeled GPS points whose predicted arc equals the reference arc (points without candidates count as wrong)."""
    m = points[points.label >= 0].merge(preds, on=["trip", "point"], how="left")
    return float((m.pred == m.label).mean())


def candidate_metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y, yhat, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(y, yhat),
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "confusion": confusion_matrix(y, yhat, labels=[0, 1]).tolist(),
    }


def collapse(seq):
    seq = np.asarray(seq)
    return seq[np.r_[True, seq[1:] != seq[:-1]]] if len(seq) else seq


def continuity(pred_arcs, arcs):
    """Fraction of arc changes in the predicted sequence that move to a directly connected arc."""
    seq = collapse(pred_arcs)
    if len(seq) < 2:
        return 1.0
    return float((arcs[seq[:-1], 1] == arcs[seq[1:], 0]).mean())


def route_prf(pred_route, ref_route):
    pred, ref = set(pred_route.tolist()), set(ref_route.tolist())
    tp = len(pred & ref)
    p = tp / max(len(pred), 1)
    r = tp / max(len(ref), 1)
    return {"route_precision": p, "route_recall": r, "route_f1": 2 * p * r / max(p + r, 1e-9)}

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.data import load_trip
from pathsnap.hmm import decode
from pathsnap.pipeline import evaluate_trip
from scripts.train_models import MOD, N_FOLDS, RES, fit, load, predict

MODEL = "LightGBM_tuned"
BETAS = [1, 2, 5, 10, 20, 50, 100]


def score_trip(args):
    part, trip_id, cand, betas = args
    trip = load_trip(Path("data", part, trip_id))
    clf = np.full(len(trip.xy), -1)
    best = cand.loc[cand.groupby("point").p.idxmax()]
    clf[best.point.to_numpy()] = best.arc.to_numpy()
    rows = []
    for beta in betas:
        hmm, route = decode(trip, cand, beta)
        res = {"trip": trip, "classifier_pred": clf, "hmm_pred": hmm, "route": route}
        rows.append({"trip": trip_id, "beta": beta, **evaluate_trip(Path("data", part, trip_id), res)})
    return rows


def run(part, cand, betas):
    jobs = [(part, t, c, betas) for t, c in cand.groupby("trip")]
    with Pool(8) as pool:
        return pd.DataFrame([r for rows in pool.map(score_trip, jobs) for r in rows])


def weighted(df, col):
    return (df[col] * df.n_labeled).sum() / df.n_labeled.sum()


def main():
    model = joblib.load(MOD / f"{MODEL}.joblib")
    train, _ = load("train")
    test, _ = load("student_test")

    # Out-of-fold probabilities so the decoder is tuned on honest (not in-sample) classifier outputs.
    oof = []
    for tr, va in GroupKFold(N_FOLDS).split(train, groups=train.trip):
        m, _ = fit(clone(model), train.iloc[tr], None)
        oof.append(predict(m, train.iloc[va])[0])
    grid = run("train", pd.concat(oof), BETAS)
    summary = grid.groupby("beta").apply(lambda g: pd.Series({
        "classifier_top1": weighted(g, "classifier_top1"), "hmm_top1": weighted(g, "hmm_top1"),
        "hmm_route_f1": g.hmm_route_f1.mean(), "hmm_continuity": g.hmm_continuity.mean()}))
    summary.to_csv(RES / "hmm_beta_grid_train.csv")
    print(summary.to_string())
    beta = float(summary.hmm_top1.idxmax())

    scored, _ = predict(model, test)
    res = run("student_test", scored, [beta])
    res.to_csv(RES / "hmm_student_test.csv", index=False)
    cols = ["top1", "continuity", "route_precision", "route_recall", "route_f1"]
    comp = pd.DataFrame({name: {c: weighted(res, f"{name}_{c}") if c == "top1" else res[f"{name}_{c}"].mean()
                                for c in cols} for name in ["classifier", "hmm"]})
    comp.loc["continuity", "reference"] = res.reference_continuity.mean()
    comp.to_csv(RES / "hmm_vs_classifier.csv")
    print(f"beta={beta}\n{comp.to_string()}")
    (MOD / "config.json").write_text(json.dumps({"model": f"{MODEL}.joblib", "beta": beta}, indent=1))

    x = np.arange(len(res))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, res.classifier_top1, 0.4, label="classifier only")
    ax.bar(x + 0.2, res.hmm_top1, 0.4, label="classifier + HMM")
    ax.set_xticks(x, res.trip, rotation=45)
    ax.set_ylim(0.6, 1)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title(f"Student-test per trip (beta={beta:g})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RES / "hmm_vs_classifier.png", dpi=150)


if __name__ == "__main__":
    main()

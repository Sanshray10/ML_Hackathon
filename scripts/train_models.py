import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.data import load_route
from pathsnap.features import FEATURES
from pathsnap.metrics import candidate_metrics, collapse, continuity, point_predictions, route_prf, top1_accuracy

SEED = 42
N_FOLDS = 5
RES, MOD = Path("results"), Path("models")

# name: (family, estimator, config description, max training rows or None)
MODELS = {
    "GaussianNB": ("Probabilistic", GaussianNB(), "default var_smoothing=1e-9", None),
    "LogReg": ("Linear", make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)),
               "L2, C=1, standardised", None),
    "KNN": ("Instance-based", make_pipeline(StandardScaler(), KNeighborsClassifier(50, weights="distance", n_jobs=-1)),
            "k=50, distance-weighted, standardised, 200k-row subsample", 200_000),
    "RandomForest": ("Ensemble (bagging)",
                     RandomForestClassifier(200, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
                     "200 trees, min_samples_leaf=5", None),
    "LightGBM": ("Boosting", LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1),
                 "defaults: 100 trees, lr=0.1, 31 leaves", None),
    "MLP": ("Neural network",
            make_pipeline(StandardScaler(), MLPClassifier((64, 32), batch_size=1024, max_iter=100, early_stopping=True,
                                                          random_state=SEED)),
            "64-32 ReLU, adam, early stopping (10% val)", None),
}


def load(part):
    cand = pd.read_pickle(f"features/{part}_candidates.pkl")
    points = pd.read_pickle(f"features/{part}_points.pkl")
    labeled = points[points.label >= 0][["trip", "point"]]
    return cand.merge(labeled, on=["trip", "point"]), points


def fit(est, cand, max_rows):
    if max_rows and len(cand) > max_rows:
        cand = cand.sample(max_rows, random_state=SEED)
    start = time.perf_counter()
    est.fit(cand[FEATURES].to_numpy(), cand.y.to_numpy())
    return est, time.perf_counter() - start


def predict(est, cand):
    start = time.perf_counter()
    p = est.predict_proba(cand[FEATURES].to_numpy())[:, 1]
    return cand.assign(p=p), time.perf_counter() - start


def sequence_metrics(scored, part):
    """Per-trip route continuity and route-level arc-set F1 of the classifier-only point predictions."""
    rows = []
    for trip, preds in point_predictions(scored).groupby("trip"):
        d = Path("data", part, trip)
        arcs = pd.read_csv(d / f"{trip}.arcs", sep=r"\s+", header=None).to_numpy()
        seq = preds.sort_values("point").pred.to_numpy()
        rows.append({"trip": trip, "continuity": continuity(seq, arcs), **route_prf(collapse(seq), load_route(d))})
    return pd.DataFrame(rows)


def main(names):
    RES.mkdir(exist_ok=True)
    MOD.mkdir(exist_ok=True)
    train, train_pts = load("train")
    test, test_pts = load("student_test")
    folds = list(GroupKFold(N_FOLDS).split(train, groups=train.trip))
    fold_rows, summary, cms = [], [], {}

    for name in names:
        family, est, config, max_rows = MODELS[name]
        print(f"== {name}", flush=True)
        for k, (tr, va) in enumerate(folds):
            m, _ = fit(est, train.iloc[tr], max_rows)
            scored, _ = predict(m, train.iloc[va])
            cm = candidate_metrics(scored.y.to_numpy(), scored.p.to_numpy())
            top1 = top1_accuracy(train_pts[train_pts.trip.isin(scored.trip.unique())], point_predictions(scored))
            fold_rows.append({"model": name, "fold": k, "f1_macro": cm["f1_macro"], "top1": top1,
                              "log_loss": cm["log_loss"]})
            print(f"  fold {k}: f1={cm['f1_macro']:.4f} top1={top1:.4f} logloss={cm['log_loss']:.4f}", flush=True)

        m, train_time = fit(est, train, max_rows)
        joblib.dump(m, MOD / f"{name}.joblib")
        train_top1 = top1_accuracy(train_pts, point_predictions(predict(m, train)[0]))
        scored, infer_time = predict(m, test)
        cm = candidate_metrics(scored.y.to_numpy(), scored.p.to_numpy())
        cms[name] = cm["confusion"]
        seq = sequence_metrics(scored, "student_test")
        seq.assign(model=name).to_csv(RES / f"seq_{name}.csv", index=False)
        cv = pd.DataFrame([r for r in fold_rows if r["model"] == name])
        summary.append({
            "model": name, "family": family, "config": config,
            "cv_f1_macro": cv.f1_macro.mean(), "cv_f1_std": cv.f1_macro.std(),
            "cv_top1": cv.top1.mean(), "cv_log_loss": cv.log_loss.mean(),
            "train_top1": train_top1,
            "test_top1": top1_accuracy(test_pts, point_predictions(scored)),
            "test_accuracy": cm["accuracy"], "test_precision_macro": cm["precision_macro"],
            "test_recall_macro": cm["recall_macro"], "test_f1_macro": cm["f1_macro"], "test_log_loss": cm["log_loss"],
            "test_continuity": seq.continuity.mean(), "test_route_f1": seq.route_f1.mean(),
            "train_time_s": train_time, "inference_time_s": infer_time,
        })
        print(pd.Series(summary[-1]).to_string(), flush=True)
        if name == "MLP":
            plot_mlp(m[-1])

    pd.DataFrame(fold_rows).to_csv(RES / "cv_folds.csv", index=False)
    pd.DataFrame(summary).to_csv(RES / "model_comparison.csv", index=False)
    (RES / "confusion_matrices.json").write_text(json.dumps(cms, indent=1))
    plot_comparison(pd.DataFrame(summary), pd.DataFrame(fold_rows), cms)


def plot_mlp(mlp):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mlp.loss_curve_, label="training loss (log loss)")
    ax2 = ax.twinx()
    ax2.plot(mlp.validation_scores_, color="tab:orange", label="validation accuracy")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training log loss")
    ax2.set_ylabel("validation accuracy")
    fig.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(RES / "mlp_training_curve.png", dpi=150)


def plot_comparison(summary, folds, cms):
    fig, ax = plt.subplots(figsize=(7, 4))
    folds.boxplot("f1_macro", by="model", ax=ax)
    ax.set_title("5-fold GroupKFold CV macro F1 (train, 80 trips)")
    fig.suptitle("")
    ax.set_ylabel("macro F1")
    fig.tight_layout()
    fig.savefig(RES / "cv_f1.png", dpi=150)

    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, summary.test_top1, 0.4, label="Top-1 accuracy")
    ax.bar(x + 0.2, summary.test_f1_macro, 0.4, label="macro F1")
    ax.set_xticks(x, summary.model)
    ax.set_ylim(0.5, 1)
    ax.set_title("Student-test (10 trips)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RES / "student_test_top1_f1.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, summary.cv_log_loss, 0.4, label="CV")
    ax.bar(x + 0.2, summary.test_log_loss, 0.4, label="student-test")
    ax.set_xticks(x, summary.model)
    ax.set_yscale("log")
    ax.set_ylabel("log loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RES / "log_loss.png", dpi=150)

    fig, axes = plt.subplots(1, len(cms), figsize=(3 * len(cms), 3))
    for ax, (name, cm) in zip(np.atleast_1d(axes), cms.items()):
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=8)
        ax.set_title(name)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlabel("predicted")
        ax.set_ylabel("reference")
    fig.tight_layout()
    fig.savefig(RES / "confusion_matrices.png", dpi=150)


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))

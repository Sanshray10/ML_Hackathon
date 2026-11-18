import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.metrics import candidate_metrics, point_predictions, top1_accuracy
from scripts.train_models import MOD, N_FOLDS, RES, SEED, fit, load, predict, sequence_metrics

N_TRIALS = 40


def space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }


def make(params):
    return LGBMClassifier(**params, random_state=SEED, n_jobs=-1, verbose=-1)


def cross_validate(params, train, train_pts, folds):
    rows = []
    for tr, va in folds:
        m, _ = fit(make(params), train.iloc[tr], None)
        scored, _ = predict(m, train.iloc[va])
        cm = candidate_metrics(scored.y.to_numpy(), scored.p.to_numpy())
        top1 = top1_accuracy(train_pts[train_pts.trip.isin(scored.trip.unique())], point_predictions(scored))
        rows.append({"top1": top1, "f1_macro": cm["f1_macro"], "log_loss": cm["log_loss"]})
    return pd.DataFrame(rows).mean()


def evaluate(name, params, train, train_pts, test, test_pts, folds):
    cv = cross_validate(params, train, train_pts, folds)
    m, train_time = fit(make(params), train, None)
    scored, infer_time = predict(m, test)
    cm = candidate_metrics(scored.y.to_numpy(), scored.p.to_numpy())
    seq = sequence_metrics(scored, "student_test")
    row = {"model": name, "cv_top1": cv.top1, "cv_f1_macro": cv.f1_macro, "cv_log_loss": cv.log_loss,
           "test_top1": top1_accuracy(test_pts, point_predictions(scored)), "test_f1_macro": cm["f1_macro"],
           "test_log_loss": cm["log_loss"], "test_continuity": seq.continuity.mean(),
           "train_time_s": train_time, "inference_time_s": infer_time}
    return m, row


def main():
    train, train_pts = load("train")
    test, test_pts = load("student_test")
    folds = list(GroupKFold(N_FOLDS).split(train, groups=train.trip))
    log = []

    def objective(trial):
        params = space(trial)
        start = time.perf_counter()
        cv = cross_validate(params, train, train_pts, folds)
        log.append({"trial": trial.number, **params, **cv.add_prefix("cv_").to_dict(),
                    "seconds": time.perf_counter() - start})
        print(f"trial {trial.number}: top1={cv.top1:.4f} logloss={cv.log_loss:.4f} {params}", flush=True)
        return cv.top1

    start = time.perf_counter()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS)
    search_time = time.perf_counter() - start
    pd.DataFrame(log).to_csv(RES / "hpo_trials.csv", index=False)

    best = {**space(optuna.trial.FixedTrial(study.best_params))}
    _, untuned = evaluate("LightGBM (untuned)", {}, train, train_pts, test, test_pts, folds)
    tuned_model, tuned = evaluate("LightGBM (tuned)", best, train, train_pts, test, test_pts, folds)
    joblib.dump(tuned_model, MOD / "LightGBM_tuned.joblib")
    comp = pd.DataFrame([untuned, tuned])
    comp.to_csv(RES / "hpo_comparison.csv", index=False)
    (RES / "hpo_best.json").write_text(json.dumps({"best_params": best, "best_cv_top1": study.best_value,
                                                   "n_trials": N_TRIALS, "search_time_s": search_time}, indent=1))
    print(comp.to_string())

    trials = pd.DataFrame(log)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(trials.trial, trials.cv_top1, "o", alpha=0.6, label="trial")
    ax.plot(trials.trial, np.maximum.accumulate(trials.cv_top1), label="best so far")
    ax.set_xlabel("trial")
    ax.set_ylabel("CV Top-1 accuracy")
    ax.set_title("Optuna TPE search, LightGBM")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RES / "hpo_history.png", dpi=150)


if __name__ == "__main__":
    main()

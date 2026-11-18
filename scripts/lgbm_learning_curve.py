import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.features import FEATURES
from scripts.train_models import N_FOLDS, RES, SEED, load

# Train/validation log loss per boosting round; validation = the trips of CV fold 0, so no trip is on both sides.
train, _ = load("train")
tr, va = next(GroupKFold(N_FOLDS).split(train, groups=train.trip))
X, y = train[FEATURES].to_numpy(), train.y.to_numpy()
best = json.loads((RES / "hpo_best.json").read_text())["best_params"]

curves = {}
for name, params in [("untuned", {}), ("tuned", best)]:
    m = LGBMClassifier(**params, random_state=SEED, n_jobs=-1, verbose=-1)
    m.fit(X[tr], y[tr], eval_X=(X[tr], X[va]), eval_y=(y[tr], y[va]), eval_names=["train", "validation"],
          eval_metric="binary_logloss")
    for split in ["train", "validation"]:
        curves[f"{name}_{split}"] = pd.Series(m.evals_result_[split]["binary_logloss"])
    v = curves[f"{name}_validation"]
    print(f"{name}: rounds={len(v)} final val logloss={v.iloc[-1]:.4f} min={v.min():.4f} at round {v.idxmin() + 1}")

out = pd.DataFrame(curves)
out.index = out.index + 1
out.index.name = "round"
out.to_csv(RES / "lgbm_training_curve.csv")

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, name in zip(axes, ["untuned", "tuned"]):
    ax.plot(out[f"{name}_train"], label="train")
    ax.plot(out[f"{name}_validation"], label="validation (held-out trips)")
    ax.set_title(f"LightGBM {name}")
    ax.set_xlabel("boosting round")
    ax.legend()
axes[0].set_ylabel("log loss")
axes[0].set_ylim(0, 0.25)
fig.tight_layout()
fig.savefig(RES / "lgbm_training_curve.png", dpi=150)

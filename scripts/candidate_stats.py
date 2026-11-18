import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_models import RES, load

rows = []
for part in ["train", "student_test"]:
    cand, points = load(part)
    labeled = points[points.label >= 0]
    hit = cand[cand.y == 1][["trip", "point"]].drop_duplicates()
    rows.append({"partition": part, "candidate_rows_all": len(pd.read_pickle(f"features/{part}_candidates.pkl")),
                 "candidate_rows_labeled": len(cand), "positive_share": cand.y.mean(),
                 "points": len(points), "labeled_points": len(labeled),
                 "candidates_per_point": cand.groupby(["trip", "point"]).size().mean(),
                 "miss_rate": 1 - len(hit) / len(labeled)})
out = pd.DataFrame(rows)
out.to_csv(RES / "candidate_stats.csv", index=False)
print(out.to_string())

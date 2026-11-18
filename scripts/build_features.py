import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.data import align_route, load_route, load_trip
from pathsnap.features import build_candidates

OUT = Path("features")


def process(trip_dir):
    trip = load_trip(trip_dir)
    label = align_route(trip, load_route(trip_dir))
    cand = build_candidates(trip)
    cand["y"] = (cand.arc.to_numpy() == label[cand.point.to_numpy()]).astype(int)
    cand.insert(0, "trip", trip.id)
    points = pd.DataFrame({"trip": trip.id, "point": range(len(label)), "label": label})
    return cand, points


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for part in ["train", "student_test"]:
        dirs = sorted(Path("data", part).iterdir())
        with Pool(6) as pool:
            res = pool.map(process, dirs)
        cand = pd.concat([c for c, _ in res], ignore_index=True)
        points = pd.concat([p for _, p in res], ignore_index=True)
        cand.to_pickle(OUT / f"{part}_candidates.pkl")
        points.to_pickle(OUT / f"{part}_points.pkl")
        labeled = points[points.label >= 0]
        hit = cand[cand.y == 1][["trip", "point"]].drop_duplicates()
        miss = 1 - len(hit) / len(labeled)
        print(f"{part}: {len(dirs)} trips, {len(points)} points, {len(labeled)} labeled, "
              f"{len(cand)} candidate rows, candidate miss rate {miss:.4f}")

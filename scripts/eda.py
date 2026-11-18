import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.data import align_route, load_route, load_trip, point_seg

OUT = Path("results/eda")


def read_raw(path):
    return pd.read_csv(path, sep=r"\s+", header=None)


def inspect(part, d):
    i = d.name
    raw = {ext: read_raw(d / f"{i}.{ext}") for ext in ["track", "nodes", "arcs", "route"]}
    track, nodes, arcs, route = raw["track"], raw["nodes"], raw["arcs"].to_numpy(), raw["route"][0].to_numpy()
    dt = np.diff(track[2].to_numpy())
    n = len(nodes)
    g = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(n, n))
    n_weak, weak = connected_components(g, directed=True, connection="weak")
    n_strong, _ = connected_components(g, directed=True, connection="strong")
    degree = np.bincount(arcs.ravel(), minlength=n)

    trip = load_trip(d)
    label = align_route(trip, route)
    a, b = trip.node_xy[trip.arcs[route, 0]], trip.node_xy[trip.arcs[route, 1]]
    to_route = np.array([point_seg(p, a, b)[0].min() for p in trip.xy[::5]])
    return {
        "partition": part, "trip": i,
        **{f"{ext}_kb": (d / f"{i}.{ext}").stat().st_size / 1024 for ext in raw},
        "n_points": len(track), "n_nodes": n, "n_arcs": len(arcs), "route_len": len(route),
        "missing_values": int(sum(r.isna().sum().sum() for r in raw.values())),
        "invalid_coords": int(((track[0].abs() > 180) | (track[1].abs() > 90)).sum()),
        "duplicate_points": int(track.duplicated([0, 1]).sum()),
        "non_increasing_t": int((dt <= 0).sum()), "dt_1s_share": float((dt == 1).mean()),
        "gaps_over_5s": int((dt > 5).sum()), "max_gap_s": float(dt.max()),
        "duration_min": float((track[2].iloc[-1] - track[2].iloc[0]) / 60),
        "lon": float(track[0].mean()), "lat": float(track[1].mean()),
        "extent_km": float(np.linalg.norm(np.ptp(trip.xy, axis=0)) / 1000),
        "weak_components": n_weak, "largest_component_share": float(np.bincount(weak).max() / n),
        "strong_components": n_strong, "isolated_nodes": int((degree == 0).sum()),
        "duplicate_arcs": int(pd.DataFrame(arcs).duplicated().sum()),
        "self_loops": int((arcs[:, 0] == arcs[:, 1]).sum()),
        "route_connected_share": float((arcs[route[:-1], 1] == arcs[route[1:], 0]).mean()),
        "route_repeated_arcs": int(len(route) - len(set(route))),
        "gps_to_route_median_m": float(np.median(to_route)), "gps_to_route_p95_m": float(np.percentile(to_route, 95)),
        "labeled_share": float((label >= 0).mean()),
    }, np.linalg.norm(np.diff(trip.xy, axis=0), axis=1) / np.maximum(dt, 1e-9), to_route


def plot_trip(ax, part, trip_id, margin=150):
    d = Path("data", part, trip_id)
    trip, route = load_trip(d), load_route(d)
    lo, hi = trip.xy.min(0) - margin, trip.xy.max(0) + margin
    a, b = trip.node_xy[trip.arcs[:, 0]], trip.node_xy[trip.arcs[:, 1]]
    near = ((a >= lo) & (a <= hi)).all(1)
    for p, q in zip(a[near], b[near]):
        ax.plot([p[0], q[0]], [p[1], q[1]], color="0.8", lw=0.5)
    for k in route:
        p, q = trip.node_xy[trip.arcs[k]]
        ax.plot([p[0], q[0]], [p[1], q[1]], color="tab:green", lw=2)
    ax.scatter(*trip.xy.T, s=1, color="tab:blue")
    ax.set_title(f"{part}/{trip_id}")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, speeds, dists = [], [], []
    for part in ["train", "student_test"]:
        for d in sorted(Path("data", part).iterdir()):
            r, s, t = inspect(part, d)
            rows.append(r)
            speeds.append(s)
            dists.append(t)
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT / "inventory.csv", index=False)
    inv.groupby("partition").describe().T.to_csv(OUT / "summary_by_partition.csv")
    pd.set_option("display.width", 200)
    print(inv.groupby("partition")[["n_points", "n_nodes", "n_arcs", "route_len", "missing_values", "invalid_coords",
                                    "gaps_over_5s", "max_gap_s", "isolated_nodes", "weak_components",
                                    "gps_to_route_median_m", "labeled_share"]].agg(["mean", "min", "max"]).T)
    print("\nanomalies (max gap > 10 min, median GPS-to-route > 30 m, or labeled share < 0.9):")
    print(inv[(inv.max_gap_s > 600) | (inv.gps_to_route_median_m > 30) | (inv.labeled_share < 0.9)]
          [["partition", "trip", "max_gap_s", "gps_to_route_median_m", "labeled_share"]].to_string(index=False))

    train = inv.partition == "train"
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].hist(inv[train].n_points, bins=30)
    axes[0, 0].set_title("GPS points per trip (train)")
    gaps = np.concatenate([np.diff(pd.read_csv(Path("data", r.partition, r.trip, f"{r.trip}.track"), sep=r"\s+",
                                               header=None)[2]) for r in inv[train].itertuples()])
    axes[0, 1].hist(np.clip(gaps, 0, 10), bins=20, log=True)
    axes[0, 1].set_title("Sampling interval (s, clipped at 10, train)")
    axes[1, 0].hist(np.clip(np.concatenate([d for d, tr in zip(dists, train) if tr]), 0, 100), bins=50)
    axes[1, 0].set_title("GPS distance to reference route (m, clipped, train)")
    axes[1, 1].hist(np.clip(np.concatenate([s for s, tr in zip(speeds, train) if tr]), 0, 60), bins=60, log=True)
    axes[1, 1].set_title("Speed between fixes (m/s, clipped, train)")
    fig.tight_layout()
    fig.savefig(OUT / "distributions.png", dpi=150)

    fig, ax = plt.subplots(figsize=(10, 5))
    for part, c in [("train", "tab:blue"), ("student_test", "tab:orange")]:
        sub = inv[inv.partition == part]
        ax.scatter(sub.lon, sub.lat, s=15, c=c, label=part)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 75)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Trip locations (mean GPS position)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "trip_locations.png", dpi=150)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, tid in zip(axes.ravel(), ["00000001", "00000005", "00000039", "00000054", "00000002", "00000046"]):
        plot_trip(ax, "train", tid)
    fig.suptitle("GPS (blue), reference route (green), road network (grey); local UTM metres")
    fig.tight_layout()
    fig.savefig(OUT / "trajectories.png", dpi=150)


if __name__ == "__main__":
    main()

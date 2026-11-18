import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .data import point_seg

RADIUS = 50.0
MAX_CANDIDATES = 8
SAMPLE_STEP = 5.0
WINDOW = 3

FEATURES = [
    "dist", "dist_rel", "rank", "n_cand", "heading_cos", "speed",
    "arc_len", "pos", "win_dist", "win_dist_rel", "win_heading_cos",
]


def nearby_arcs(trip):
    """Arcs with any part within RADIUS of the track, found via points sampled every SAMPLE_STEP metres along each arc."""
    a, b = trip.node_xy[trip.arcs[:, 0]], trip.node_xy[trip.arcs[:, 1]]
    lo, hi = trip.xy.min(0) - RADIUS, trip.xy.max(0) + RADIUS
    keep = np.flatnonzero(((np.maximum(a, b) >= lo) & (np.minimum(a, b) <= hi)).all(1))
    n = np.ceil(np.linalg.norm(b[keep] - a[keep], axis=1) / SAMPLE_STEP).astype(int) + 1
    arc_of = np.repeat(keep, n)
    frac = np.concatenate([np.linspace(0, 1, k) for k in n]) if len(n) else np.empty(0)
    pts = a[arc_of] + frac[:, None] * (b[arc_of] - a[arc_of])
    return cKDTree(pts), arc_of


def headings(trip, step=2):
    """Unit direction of travel at each point (from neighbours step points away), displacement and speed."""
    i = np.arange(len(trip.xy))
    fwd, bwd = np.minimum(i + step, len(i) - 1), np.maximum(i - step, 0)
    v = trip.xy[fwd] - trip.xy[bwd]
    disp = np.linalg.norm(v, axis=1)
    speed = disp / np.maximum(trip.t[fwd] - trip.t[bwd], 1e-3)
    return v / np.maximum(disp, 1e-9)[:, None], disp, speed


def build_candidates(trip):
    """One row per (GPS point, candidate arc): the MAX_CANDIDATES nearest arcs within RADIUS, with features."""
    tree, arc_of = nearby_arcs(trip)
    hits = tree.query_ball_point(trip.xy, RADIUS + SAMPLE_STEP)
    point = np.repeat(np.arange(len(hits)), [len(h) for h in hits])
    arc = arc_of[np.concatenate(hits).astype(int)] if len(point) else np.empty(0, int)
    pairs = np.unique(np.column_stack([point, arc]), axis=0)
    point, arc = pairs[:, 0], pairs[:, 1]

    a, b = trip.node_xy[trip.arcs[arc, 0]], trip.node_xy[trip.arcs[arc, 1]]
    dist, pos = point_seg(trip.xy[point], a, b)
    df = pd.DataFrame({"point": point, "arc": arc, "dist": dist, "pos": pos})
    df = df[df.dist <= RADIUS].sort_values(["point", "dist"])
    df["rank"] = df.groupby("point").cumcount()
    df = df[df["rank"] < MAX_CANDIDATES].reset_index(drop=True)
    return add_features(trip, df)


def add_features(trip, df):
    point, arc = df.point.to_numpy(), df.arc.to_numpy()
    a, b = trip.node_xy[trip.arcs[arc, 0]], trip.node_xy[trip.arcs[arc, 1]]
    seg = b - a
    arc_len = np.linalg.norm(seg, axis=1)
    arc_dir = seg / np.maximum(arc_len, 1e-9)[:, None]
    head, disp, speed = headings(trip)

    moving = disp[point] > 2.0
    df["heading_cos"] = np.where(moving, (head[point] * arc_dir).sum(1), 0.0)
    df["speed"] = speed[point]
    df["arc_len"] = arc_len
    df["dist_rel"] = df.dist - df.groupby("point").dist.transform("min")
    df["n_cand"] = df.groupby("point").arc.transform("size")

    # Context from neighbouring GPS points: how well this arc explains the local stretch of track.
    offsets = np.arange(-WINDOW, WINDOW + 1)
    nb = np.clip(point[:, None] + offsets, 0, len(trip.xy) - 1)
    df["win_dist"] = point_seg(trip.xy[nb], a[:, None], b[:, None])[0].mean(1)
    df["win_dist_rel"] = df.win_dist - df.groupby("point").win_dist.transform("min")
    hc = (head[nb] * arc_dir[:, None]).sum(-1)
    df["win_heading_cos"] = np.where(disp[nb] > 2.0, hc, 0.0).mean(1)
    return df

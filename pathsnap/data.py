from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyproj import Transformer

LABEL_MAX_DIST = 50.0


@dataclass
class Trip:
    id: str
    lonlat: np.ndarray
    t: np.ndarray
    xy: np.ndarray
    node_lonlat: np.ndarray
    node_xy: np.ndarray
    arcs: np.ndarray


def utm_epsg(lon, lat):
    zone = int((lon + 180) // 6) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


def load_trip(trip_dir):
    d = Path(trip_dir)
    track = np.loadtxt(d / f"{d.name}.track", ndmin=2)
    nodes = np.loadtxt(d / f"{d.name}.nodes", ndmin=2)
    arcs = np.loadtxt(d / f"{d.name}.arcs", dtype=np.int64, ndmin=2)
    # Trips are spread worldwide, so each trip gets its own UTM zone.
    tr = Transformer.from_crs(4326, utm_epsg(*track[:, :2].mean(0)), always_xy=True)
    xy = np.column_stack(tr.transform(track[:, 0], track[:, 1]))
    node_xy = np.column_stack(tr.transform(nodes[:, 0], nodes[:, 1]))
    return Trip(d.name, track[:, :2], track[:, 2], xy, nodes, node_xy, arcs)


def load_route(trip_dir):
    d = Path(trip_dir)
    return np.loadtxt(d / f"{d.name}.route", dtype=np.int64, ndmin=1)


def point_seg(p, a, b):
    """Distance from points p to segments a-b (broadcasting), plus position t in [0, 1] along the segment."""
    ab = b - a
    t = np.clip(((p - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0, 1)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1), t


def align_route(trip, route):
    """Assign each GPS point a reference arc: monotonic (order-preserving) alignment of the track to the route
    minimising total point-to-arc distance. Points farther than LABEL_MAX_DIST from their aligned arc get -1."""
    a = trip.node_xy[trip.arcs[route, 0]]
    b = trip.node_xy[trip.arcs[route, 1]]
    n, m = len(trip.xy), len(route)
    back = np.empty((n, m), dtype=np.int32)
    idx = np.arange(m)
    cost, _ = point_seg(trip.xy[0], a, b)
    for i in range(1, n):
        pm = np.minimum.accumulate(cost)
        back[i] = np.maximum.accumulate(np.where(cost == pm, idx, 0))
        cost = pm + point_seg(trip.xy[i], a, b)[0]
    j = np.empty(n, dtype=np.int64)
    j[-1] = cost.argmin()
    for i in range(n - 1, 0, -1):
        j[i - 1] = back[i, j[i]]
    dist = point_seg(trip.xy, a[j], b[j])[0]
    return np.where(dist <= LABEL_MAX_DIST, route[j], -1)

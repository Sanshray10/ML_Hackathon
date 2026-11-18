import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

BETA = 10.0
UNREACHABLE = -30.0
GAP_SECONDS = 60.0
GRAPH_MARGIN = 300.0
SP_LIMIT = 2000.0


def arc_lengths(trip):
    return np.linalg.norm(trip.node_xy[trip.arcs[:, 1]] - trip.node_xy[trip.arcs[:, 0]], axis=1)


class RoadGraph:
    """Directed road graph restricted to nodes within GRAPH_MARGIN of the track (plus all candidate arcs' nodes)."""

    def __init__(self, trip, keep_arcs):
        arcs = trip.arcs
        near, _ = cKDTree(trip.xy).query(trip.node_xy, distance_upper_bound=GRAPH_MARGIN)
        keep = np.isfinite(near)
        keep[arcs[keep_arcs].ravel()] = True
        self.local = np.full(len(keep), -1)
        self.local[keep] = np.arange(keep.sum())
        idx = np.flatnonzero(keep[arcs[:, 0]] & keep[arcs[:, 1]] & (arcs[:, 0] != arcs[:, 1]))
        e = pd.DataFrame({"u": self.local[arcs[idx, 0]], "v": self.local[arcs[idx, 1]],
                          "w": arc_lengths(trip)[idx] + 1e-3, "arc": idx})
        e = e.sort_values("w").drop_duplicates(["u", "v"])
        n = int(keep.sum())
        self.graph = csr_matrix((e.w, (e.u, e.v)), shape=(n, n))
        self.edge_arc = dict(zip(zip(e.u, e.v), e.arc))

    def distances(self, src, dst):
        s, d = self.local[src], self.local[dst]
        out = np.empty((len(s), len(d)), dtype=np.float32)
        for i in range(0, len(s), 256):
            out[i:i + 256] = dijkstra(self.graph, indices=s[i:i + 256], limit=SP_LIMIT)[:, d]
        return out

    def path_arcs(self, u, v):
        src, dst = self.local[u], self.local[v]
        dist, pred = dijkstra(self.graph, indices=src, limit=SP_LIMIT, return_predecessors=True)
        if not np.isfinite(dist[dst]):
            return []
        path = []
        while dst != src:
            path.append(self.edge_arc[(pred[dst], dst)])
            dst = pred[dst]
        return path[::-1]


def decode(trip, cand, beta=BETA):
    """Viterbi over candidate arcs. cand: one trip's candidate rows with point, arc, pos, p.
    Returns the decoded arc per GPS point (-1 where a point has no candidates) and a connected route."""
    cand = cand.sort_values(["point", "rank"])
    length = arc_lengths(trip)
    uniq = np.unique(cand.arc.to_numpy())
    graph = RoadGraph(trip, uniq)
    sp = graph.distances(trip.arcs[uniq, 1], trip.arcs[uniq, 0])

    point = cand.point.to_numpy()
    starts = np.flatnonzero(np.r_[True, point[1:] != point[:-1]])
    bounds = np.r_[starts, len(point)]
    arc, ui = cand.arc.to_numpy(), np.searchsorted(uniq, cand.arc.to_numpy())
    pos, emis = cand.pos.to_numpy(), np.log(np.clip(cand.p.to_numpy(), 1e-6, 1))

    pred = np.full(len(trip.xy), -1)
    pts = point[starts]
    # Start a fresh chain after long GPS gaps: the vehicle may have moved anywhere in between.
    breaks = np.r_[0, np.flatnonzero(np.diff(trip.t[pts]) > GAP_SECONDS) + 1, len(pts)]
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        score, backs = emis[bounds[lo]:bounds[lo + 1]], []
        for k in range(lo + 1, hi):
            a, b = slice(bounds[k - 1], bounds[k]), slice(bounds[k], bounds[k + 1])
            gc = np.linalg.norm(trip.xy[pts[k]] - trip.xy[pts[k - 1]])
            la = length[arc[a]][:, None]
            route = (1 - pos[a])[:, None] * la + sp[np.ix_(ui[a], ui[b])] + (pos[b] * length[arc[b]])[None, :]
            same = arc[a][:, None] == arc[b][None, :]
            route = np.where(same, np.abs(pos[b][None, :] - pos[a][:, None]) * la, route)
            trans = np.where(np.isfinite(route), -np.abs(route - gc) / beta, UNREACHABLE)
            total = score[:, None] + trans
            backs.append(total.argmax(0))
            score = total.max(0) + emis[b]
        j = int(score.argmax())
        for k in range(hi - 1, lo, -1):
            pred[pts[k]] = arc[bounds[k] + j]
            j = backs[k - lo - 1][j]
        pred[pts[lo]] = arc[bounds[lo] + j]
    return pred, connect(trip, graph, pred)


def connect(trip, graph, pred):
    """Collapse per-point arcs into a route and fill non-adjacent consecutive arcs with shortest paths."""
    seq = pred[pred >= 0]
    seq = seq[np.r_[True, seq[1:] != seq[:-1]]] if len(seq) else seq
    route = list(seq[:1])
    for a, b in zip(seq[:-1], seq[1:]):
        if trip.arcs[a, 1] != trip.arcs[b, 0]:
            route += graph.path_arcs(trip.arcs[a, 1], trip.arcs[b, 0])
        route.append(b)
    return np.array(route, dtype=np.int64)

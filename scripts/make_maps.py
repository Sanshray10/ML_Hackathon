import json
import sys
from pathlib import Path

import folium
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.data import load_route
from pathsnap.metrics import collapse
from pathsnap.pipeline import predict_trip

OUT = Path("results/maps")
LEGEND = """<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px 12px;
border:1px solid #999;border-radius:4px;font-size:13px;line-height:1.7">
<b>Trip {trip}</b><br>
<span style="color:#1f6feb">&#9679;</span> GPS observations<br>
<span style="color:#999">&#9473;</span> Road network (near track)<br>
<span style="color:#2a9d3f">&#9473;</span> Reference route<br>
<span style="color:#f4a259">&#9473;</span> Classifier-only route<br>
<span style="color:#e5383b">&#9473;</span> Predicted route (classifier + HMM)</div>"""


def arc_lines(trip, arcs):
    nodes = trip.node_lonlat[:, ::-1]
    return [[nodes[u].tolist(), nodes[v].tolist()] for u, v in trip.arcs[arcs]]


def trip_map(trip_dir, model, beta):
    res = predict_trip(trip_dir, model, beta)
    trip = res["trip"]
    m = folium.Map()
    m.fit_bounds([trip.lonlat[:, ::-1].min(0).tolist(), trip.lonlat[:, ::-1].max(0).tolist()])

    a, b = trip.node_xy[trip.arcs[:, 0]], trip.node_xy[trip.arcs[:, 1]]
    lo, hi = trip.xy.min(0) - 100, trip.xy.max(0) + 100
    near = np.flatnonzero(((a >= lo) & (a <= hi) & (b >= lo) & (b <= hi)).all(1))
    layers = [
        ("Road network", arc_lines(trip, near), dict(color="#999", weight=1, opacity=0.6)),
        ("Reference route", arc_lines(trip, load_route(trip_dir)), dict(color="#2a9d3f", weight=7, opacity=0.6)),
        ("Classifier-only route", arc_lines(trip, collapse(res["classifier_pred"][res["classifier_pred"] >= 0])),
         dict(color="#f4a259", weight=3, opacity=0.9, dash_array="6")),
        ("Predicted route (HMM)", arc_lines(trip, res["route"]), dict(color="#e5383b", weight=3, opacity=0.9)),
    ]
    for name, lines, style in layers:
        folium.PolyLine(lines, tooltip=name, **style).add_to(folium.FeatureGroup(name, show=name != "Classifier-only route").add_to(m))

    gps = folium.FeatureGroup("GPS observations").add_to(m)
    for (lat, lon), t in zip(trip.lonlat[:, ::-1], trip.t):
        folium.CircleMarker([lat, lon], radius=2, color="#1f6feb", fill=True, weight=1, tooltip=f"t={t:g}s").add_to(gps)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND.format(trip=trip.id)))
    return m


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path("models/config.json").read_text())
    model = joblib.load(Path("models") / config["model"])
    part = sys.argv[1] if len(sys.argv) > 1 else "student_test"
    ids = sys.argv[2:] or sorted(p.name for p in Path("data", part).iterdir())
    for tid in ids:
        trip_map(Path("data", part, tid), model, config["beta"]).save(OUT / f"{part}_{tid}.html")
        print(f"saved {OUT / f'{part}_{tid}.html'}")

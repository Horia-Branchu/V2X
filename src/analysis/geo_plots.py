from pathlib import Path
import math
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import sumolib


def find_latest_csv(root_dir: Path, filename="vehicles.csv"):
    """Search for the newest matching CSV in the project folder"""

    print(f"Correlation Map for {root_dir}\n"
          f"Correlation Map is generating")

    candidates = list(root_dir.rglob(filename))
    if len(candidates) == 0:
        return None

    ###Sorting files by the last modification time
    candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return candidates[0]

def find_latest_sumocfg(root_dir, pattern="*.sumocfg"):
    """Find the most recently modified SUMO config."""
    return find_latest_csv(root_dir, pattern)


def _find_net_path_from_sumocfg(sumo_cfg_path):
    """Extract the net-file path from a SUMO .sumocfg file."""
    tree = ET.parse(sumo_cfg_path)
    root = tree.getroot()
    node = root.find(".//input/net-file")
    if node is None:
        raise FileNotFoundError("no net-file found in SUMO config.")
    net_rel = node.get("value")
    net_path = (sumo_cfg_path.parent / net_rel).resolve()
    if not net_path.exists():
        raise FileNotFoundError(f"network file not found: {net_path}")
    return net_path


def _net_bbox_from_shapes(net):
    """Compute bounding box from edge shapes."""
    xs, ys = [], []
    for e in net.getEdges():
        shp = e.getShape()
        if not shp:
            continue
        for x, y in shp:
            xs.append(x)
            ys.append(y)
    if not xs:
        return (0, 0, 1000, 1000)
    return (min(xs), min(ys), max(xs), max(ys))


def _is_real_edge(edge):
    """Ignore internal edges like ':cluster_0'."""
    return not edge.getID().startswith(':')


def compute_edge_min_speed(df):
    """Compute minimum speed and sample count per edge."""
    if not {"edge", "speed"}.issubset(df.columns):
        raise ValueError("DataFrame must have columns: edge, speed")

    d = df.copy()
    d["edge"] = d["edge"].astype(str)
    d = d[d["edge"].str.len() > 0]

    gb = d.groupby("edge")["speed"]
    out = gb.min().rename("min_speed").to_frame()
    out["hits"] = gb.size()
    out = out[out["min_speed"] > 0].reset_index()
    return out


def plot_min_speed_map(df, sumo_config, out_path, background_path=None, cmap_name="viridis", linewidth_base=1.6, linewidth_by_hits=True, hits_scale=0.45, top_n_labels=0, title="Minimum speed per edge"):
    """Draw the SUMO network colored by per-edge minimum speed."""
    stats = compute_edge_min_speed(df)
    metric_by_edge = dict(zip(stats["edge"], stats["min_speed"]))
    hits_by_edge = dict(zip(stats["edge"], stats["hits"]))

    if not metric_by_edge:
        print("No edge metrics found.")
        return

    net_path = _find_net_path_from_sumocfg(Path(sumo_config))
    net = sumolib.net.readNet(str(net_path))
    xmin, ymin, xmax, ymax = _net_bbox_from_shapes(net)

    vals = list(metric_by_edge.values())
    vmin = 0.0
    vmax = 20.0
    cmap = mpl.colormaps.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12, 10))

    if background_path and Path(background_path).exists():
        img = plt.imread(background_path)
        ax.imshow(img, extent=[xmin, xmax, ymin, ymax], origin="lower", alpha=0.65)

    labels = [] if top_n_labels > 0 else None

    for e in net.getEdges():
        if not _is_real_edge(e):
            continue
        eid = e.getID()
        if eid not in metric_by_edge:
            continue
        shp = e.getShape()
        if not shp or len(shp) < 2:
            continue

        val = metric_by_edge[eid]
        color = cmap(norm(val))
        lw = linewidth_base
        if linewidth_by_hits:
            hits = max(hits_by_edge.get(eid, 1), 1)
            lw += hits_scale * math.log10(hits)

        xs, ys = zip(*shp)
        ax.plot(xs, ys, color=color, linewidth=lw, solid_capstyle="round", alpha=0.95)

        if labels is not None:
            mid = len(xs) // 2
            labels.append((val, xs[mid], ys[mid], eid))

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.032, pad=0.02)
    cbar.set_label("Minimum speed (m/s)")

    if labels:
        labels.sort(key=lambda t: t[0])
        for val, x, y, eid in labels[:top_n_labels]:
            ax.text(x, y, f"{eid}\n{val:.1f} m/s", fontsize=8, ha="center", va="center")

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved geographic plot to: {out_path}")


def main():
    """Find latest vehicles.csv and latest .sumocfg, then generate the geo plot."""
    project_root = Path(__file__).resolve().parents[2]
    csv_path = find_latest_csv(project_root, "vehicles.csv")
    sumo_cfg_path = find_latest_sumocfg(project_root / "config", "*.sumocfg")

    if not csv_path or not csv_path.exists():
        print(f"No vehicles.csv found.")
        return
    if not sumo_cfg_path or not sumo_cfg_path.exists():
        print(f"No .sumocfg found")
        return

    print(f"Using dataset: {csv_path}\nWaiting for the plots")
    df = pd.read_csv(csv_path, low_memory=False)
    out_dir = csv_path.parent

    plot_min_speed_map(df, sumo_cfg_path, out_dir / "min_speed_map.png", None, "viridis", 1.6, True, 0.45, 0)
    print(f"All geo plots are done")


if __name__ == "__main__":
    main()

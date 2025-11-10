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


def net_bounding_box(net):
    """Compute bounding box from edge shapes."""
    all_x = []
    all_y = []

    for edge in net.getEdges():
        shape = edge.getShape()
        if shape is None or len(shape) == 0:
            continue
        for point in shape:
            x, y = point
            all_x.append(x)
            all_y.append(y)

    if len(all_x) == 0 or len(all_y) == 0:
        return (0, 0, 1000, 1000)

    x_min = min(all_x)
    y_min = min(all_y)
    x_max = max(all_x)
    y_max = max(all_y)

    return (x_min, y_min, x_max, y_max)


def _is_real_edge(edge):
    """Ignore internal edges like ':cluster_0'."""
    return not edge.getID().startswith(':')


def compute_edge_min_speed(df):
    """Compute minimum speed and sample count per edge."""
    if not {"edge", "speed"}.issubset(df.columns):
        print(f"No edge or speed columns found")
        return None

    d = df.copy()
    d["edge"] = d["edge"].astype(str)
    d = d[d["edge"].str.len() > 0]

    gb = d.groupby("edge")["speed"]
    out = gb.min().rename("min_speed").to_frame()
    out["hits"] = gb.size()
    out = out[out["min_speed"] > 0].reset_index()
    return out


def plot_min_speed_map(df, sumo_config, out_path, background_path=None,
                       cmap_name="viridis", linewidth_base=1.6, linewidth_by_hits=True,
                       hits_scale=0.45, top_n_labels=0, title="Minimum speed per edge Before V2X"):

    # Compute minimum speed for each edge from the DataFrame
    stats = compute_edge_min_speed(df)
    metric_by_edge = dict(zip(stats["edge"], stats["min_speed"]))
    hits_by_edge = dict(zip(stats["edge"], stats["hits"]))

    # stop early if no valid data was found
    if len(metric_by_edge) == 0:
        print("No edge metrics found.")
        return

    # Load SUMO network (.net.xml) from config file
    net_path = _find_net_path_from_sumocfg(Path(sumo_config))
    net = sumolib.net.readNet(str(net_path))
    xmin, ymin, xmax, ymax = net_bounding_box(net)

    # Set color normalization and colormap range
    vmin = 2
    vmax = 20.0
    cmap = mpl.colormaps.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12, 10))

    # checking if a backgroung img exists
    if background_path is not None:
        if Path(background_path).exists():
            img = plt.imread(background_path)
            ax.imshow(img, extent=[xmin, xmax, ymin, ymax], origin="lower", alpha=0.65)

    # Prepare label list only if we want to display some labels
    if top_n_labels > 0:
        labels = []
    else:
        labels = None

    # Loop through each edge in the SUMO network
    for e in net.getEdges():
        # Skip internal or artificial edges
        if _is_real_edge(e) == False:
            continue

        eid = e.getID()

        # Skip edges that do not appear in the computed stats
        if eid not in metric_by_edge:
            continue

        # Get the edge geometry
        shp = e.getShape()
        if shp is None:
            continue
        if len(shp) < 2:
            continue

        # Compute color and line width
        val = metric_by_edge[eid]
        color = cmap(norm(val))
        lw = linewidth_base

        # Adjust line thickness based on traffic intensity (hits)
        if linewidth_by_hits is True:
            hits = hits_by_edge.get(eid, 1)
            if hits < 1:
                hits = 1
            lw = lw + hits_scale * math.log10(hits)

        # extract x & y from shape
        xs, ys = zip(*shp)

        # draw the edge line on the map
        ax.plot(xs, ys, color=color, linewidth=lw, solid_capstyle="round", alpha=0.95)

        # Save a few labels for annotation if needed
        if labels is not None:
            mid = len(xs) // 2
            labels.append((val, xs[mid], ys[mid], eid))

    # Creating colorbar
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.032, pad=0.02)
    cbar.set_label("Minimum speed (m/s)")

    # Add labels directly on the map for top edges
    if labels is not None:
        labels.sort(key=lambda t: t[0])
        for val, x, y, eid in labels[:top_n_labels]:
            ax.text(x, y, f"{eid}\n{val:.1f} m/s", fontsize=8, ha="center", va="center")

    # Final map layout and styling
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    plt.tight_layout()
    out_path = Path(out_path)
    out_dir = out_path.parent
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Save figure to file
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved geographic plot to: {out_path}")


def main():
    project_root = Path(__file__).resolve().parents[2]
    csv_path = find_latest_csv(project_root, "vehicles.csv")
    sumo_cfg_path = find_latest_sumocfg(project_root / "config", "*.sumocfg")

    if not csv_path or not csv_path.exists():
        print(f"No vehicles.csv found.")
        return None
    if not sumo_cfg_path or not sumo_cfg_path.exists():
        print(f"No sumocfg found")
        return None

    print(f"Using dataset: {csv_path}\nWaiting for the plots")
    df = pd.read_csv(csv_path, low_memory=False)
    out_dir = csv_path.parent

    plot_min_speed_map(df, sumo_cfg_path, out_dir / "min_speed_map.png", None, "viridis",
                       1.6, True, 0.45, 0)
    print(f"All geo plots are done")


if __name__ == "__main__":
    main()

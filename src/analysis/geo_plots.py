import math
import xml.etree.ElementTree as ET
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import sumolib
import logging

from data_collector import baseline_filename, v2x_filename, data_dir_name
from pathlib import Path

logger = logging.getLogger("v2x")

def _find_net_path_from_sumocfg(sumo_cfg_path: Path) -> Path:
    """Extract the net-file path from a SUMO .sumocfg file."""
    tree = ET.parse(sumo_cfg_path)
    root = tree.getroot()
    node = root.find(".//input/net-file")
    if node is None:
        raise FileNotFoundError("No <net-file> found in SUMO config.")
    net_rel = node.get("value")
    net_path = (sumo_cfg_path.parent / net_rel).resolve()
    if not net_path.exists():
        raise FileNotFoundError(f"Network file not found: {net_path}")
    return net_path


def _net_bbox_from_shapes(net):
    """Compute bounding box from edge shapes."""
    xs, ys = [], []
    for e in net.getEdges():
        shp = e.getShape()
        if not shp:
            continue
        for x, y in shp:
            xs.append(x);
            ys.append(y)
    if not xs:
        raise ValueError("Network dataset has no valid edge shapes for plotting.")
    return (min(xs), min(ys), max(xs), max(ys))


def _is_real_edge(edge):
    """Ignore internal edges like ':cluster_0'."""
    return not edge.getID().startswith(':')


def compute_edge_min_speed(df: pd.DataFrame):
    """Compute minimum speed and sample count per edge."""
    if not {"edge", "speed"}.issubset(df.columns):
        raise ValueError("DataFrame must have columns: edge, speed")

    d = df.copy()
    d = df[df["speed"] > 0].copy()

    gb = d.groupby("edge")["speed"]
    out = gb.min().rename("min_speed").to_frame()
    out["hits"] = gb.size()
    return out.reset_index()

def plot_min_speed_map(
    df: pd.DataFrame,
    sumo_config: Path,
    out_path: Path,
    background_path: Path | None = None,
    cmap_name: str = "viridis",
    linewidth_base: float = 1.6,
    linewidth_by_hits: bool = True,
    hits_scale: float = 0.45,
    top_n_labels: int = 0,
    title: str = "Minimum speed per edge (geographic)"
):
    """Draw the SUMO network colored by per-edge minimum speed."""
    stats = compute_edge_min_speed(df)
    metric_by_edge = dict(zip(stats["edge"], stats["min_speed"]))
    hits_by_edge = dict(zip(stats["edge"], stats["hits"]))

    if not metric_by_edge:
        print("No edge metrics found.")
        return

    net_path = _find_net_path_from_sumocfg(sumo_config)
    net = sumolib.net.readNet(str(net_path))
    xmin, ymin, xmax, ymax = _net_bbox_from_shapes(net)

    vmin, vmax = 3, 12
    cmap = mpl.colormaps.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12, 10))

    if background_path and Path(background_path).exists():
        img = plt.imread(background_path)
        ax.imshow(
            img,
            extent=[xmin, xmax, ymin, ymax],
            origin="lower",
            alpha=0.65,
            zorder=0,
        )

    for e in net.getEdges():
        if not _is_real_edge(e):
            continue

        shp = e.getShape()
        if not shp or len(shp) < 2:
            continue

        xs, ys = zip(*shp)

        ax.plot(
            xs,
            ys,
            color="lightgray",
            linewidth=0.6,
            alpha=0.5,
            solid_capstyle="round",
            zorder=1,
        )

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
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=lw,
            solid_capstyle="round",
            alpha=0.95,
            zorder=2,
        )

        if labels is not None:
            mid = len(xs) // 2
            labels.append((val, xs[mid], ys[mid], eid))

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.032, pad=0.02)
    cbar.set_label("Minimum speed (m/s)")

    if labels:
        labels.sort(key=lambda t: t[0])  # slowest first
        for val, x, y, eid in labels[:top_n_labels]:
            ax.text(x,y,f"{eid}\n{val:.1f} m/s",fontsize=8,ha="center",va="center",zorder=3,)

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
    logger.info(f"Saved geographic plot to: {out_path}")


def generate_geo_plot(parquet_path: Path, sumo_cfg_path: Path, output_name: str):
    """load parquet and generate a geographic min-speed plot."""
    df = pd.read_parquet(parquet_path)
    out_dir = parquet_path.parent
    out_path = out_dir / output_name

    plot_min_speed_map(
        df=df,
        sumo_config=sumo_cfg_path,
        out_path=out_path,
        background_path=None,
        top_n_labels=0,
        title=f"Minimum Speed Map – {output_name.replace('.png', '')}"
    )


def compare_geo_plots(data_dir: Path):
    """Create a side-by-side comparison image of baseline vs V2X."""
    import matplotlib.image as mpimg

    baseline_img = data_dir / "min_speed_baseline.png"
    v2x_img = data_dir / "min_speed_V2X.png"

    if not baseline_img.exists() or not v2x_img.exists():
        print("Skipping comparison: one of the images does not exist.")
        return

    img_base = mpimg.imread(baseline_img)
    img_v2x = mpimg.imread(v2x_img)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    axes[0].imshow(img_base)
    axes[0].set_title("Baseline")
    axes[0].axis("off")

    axes[1].imshow(img_v2x)
    axes[1].set_title("V2X")
    axes[1].axis("off")

    out_path = data_dir / "min_speed_comparison.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

    print(f"Saved side-by-side comparison: {out_path}")


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / data_dir_name

    baseline_path = data_dir / baseline_filename
    v2x_path = data_dir / v2x_filename

    sumo_cfg_path = project_root / "config" / "simulation.sumocfg"
    if not sumo_cfg_path.exists():
        raise FileNotFoundError(f"Expected config file not found: {sumo_cfg_path}")

    print(f"Using SUMO config: {sumo_cfg_path}")
    print(f"Looking for parquet files in: {data_dir}")

    if v2x_path.exists():
        print("Generating geographic plot for V2X run...")
        generate_geo_plot(v2x_path, sumo_cfg_path, "min_speed_V2X.png")
    else:
        print("No {vehicle_filename} — skipping V2X plot")

    if baseline_path.exists():
        print("Generating geographic plot for BASELINE run...")
        generate_geo_plot(baseline_path, sumo_cfg_path, "min_speed_baseline.png")
    else:
        print(f"No {Path(baseline_filename).stem}_baseline.parquet found — skipping baseline plot")

    if v2x_path.exists() and baseline_path.exists():
        try:
            compare_geo_plots(data_dir)
        except Exception as e:
            print(f"Failed to generate comparison image: {e}")

    print("All geo plots finished.")


if __name__ == "__main__":
    main()
import math
import xml.etree.ElementTree as ET
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import sumolib
import logging
import numpy as np

from data_collector import baseline_filename, v2x_filename, data_dir_name, rl_filename
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
            xs.append(x)
            ys.append(y)
    if not xs:
        raise ValueError("Network dataset has no valid edge shapes for plotting.")
    return min(xs), min(ys), max(xs), max(ys)


def _is_real_edge(edge):
    """Ignore internal edges like ':cluster_0'."""
    return not edge.getID().startswith(':')


def compute_edge_co2_per_meter(df: pd.DataFrame, edge_lengths: dict):
    """Compute cumulative CO2 per meter and sample count per edge."""
    if not {"edge", "co2"}.issubset(df.columns):
        raise ValueError("DataFrame must have columns: edge, co2")

    gb = df.groupby("edge")["co2"]

    out = gb.sum().rename("total_co2").to_frame()
    out["edge_length"] = out.index.map(edge_lengths)
    out = out.dropna(subset=["edge_length"])
    out["co2_per_m"] = out["total_co2"] / out["edge_length"]
    out["hits"] = gb.size()

    return out.reset_index()


def compute_edge_pollution_co2(df_base: pd.DataFrame, df_v2x: pd.DataFrame, edge_lengths: dict):
    """Relative CO2 Pollution Change per meter"""
    b = compute_edge_co2_per_meter(df_base, edge_lengths)
    v = compute_edge_co2_per_meter(df_v2x, edge_lengths)

    m = v.merge(b, on="edge", suffixes=("_v2x", "_base"))
    eps = pow(10, -9)
    m["pollution_co2"] = (m["co2_per_m_v2x"]-m["co2_per_m_base"])/(m["co2_per_m_base"] + eps)
    vals = m["pollution_co2"].to_numpy()
    noise_threshold = np.percentile(np.abs(vals), 10)
    m.loc[np.abs(m["pollution_co2"]) < noise_threshold, "pollution_co2"] = 0.0

    #cap extreme ratios for stability
    m["pollution_co2"] = m["pollution_co2"].clip(-1.0, 1.0)

    m["hits"] = m["hits_v2x"] + m["hits_base"]
    return m


def plot_co2_pollution_map(
    df_pollution: pd.DataFrame,
    sumo_config: Path,
    out_path: Path,
    linewidth_base: float = 1.6,
    linewidth_by_hits: bool = True,
    hits_scale: float = 0.45,
    title: str = "CO2 Change per Edge (V2X − Baseline)"
):
    """Create a side-by-side comparison image of baseline vs V2X."""
    pollution_by_edge = dict(zip(df_pollution["edge"], df_pollution["pollution_co2"]))
    hits_by_edge = dict(zip(df_pollution["edge"], df_pollution["hits"]))

    if not pollution_by_edge:
        print(f"No CO2 pollution found.")
        return

    net_path = _find_net_path_from_sumocfg(sumo_config)
    net = sumolib.net.readNet(str(net_path))
    xmin, ymin, xmax, ymax = _net_bbox_from_shapes(net)

    # symmetric normalization
    vals = np.array(list(pollution_by_edge.values()))
    p = np.percentile(np.abs(vals), 90)
    if p == 0:
        p = 1e-6

    norm = mpl.colors.TwoSlopeNorm(vmin=-p, vcenter=0.0, vmax=p)
    cmap = mpl.colormaps.get_cmap("RdYlGn_r")

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(18, 10))

    for e in net.getEdges():
        if not _is_real_edge(e):
            continue
        shp = e.getShape()
        if not shp or len(shp) < 2:
            continue
        xs, ys = zip(*shp)
        ax_l.plot(xs, ys, color="darkgray", linewidth=0.8, alpha=0.8)

    ax_l.set_title("Baseline reference pollution map")
    ax_l.set_xlim([xmin, xmax])
    ax_l.set_ylim([ymin, ymax])
    ax_l.set_aspect("equal", adjustable="box")
    ax_l.axis("off")
    ax_l.set_xticks([])
    ax_l.set_yticks([])

    for e in net.getEdges():
        if not _is_real_edge(e):
            continue

        eid = e.getID()
        shp = e.getShape()
        if not shp or len(shp) < 2:
            continue

        xs, ys = zip(*shp)

        if eid not in pollution_by_edge:
            ax_r.plot(xs, ys, color="lightgray", linewidth=0.6, alpha=0.5)
            continue

        val = pollution_by_edge[eid]
        color = cmap(norm(val))
        lw = linewidth_base

        if linewidth_by_hits:
            hits = max(hits_by_edge.get(eid, 1), 1)
            lw += hits_scale * math.log10(hits)

        ax_r.plot(xs, ys, color=color, linewidth=lw, alpha=0.97)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_r, fraction=0.035, pad=0.02)
    cbar.set_label("")
    ax_r.set_title(title)
    ax_r.set_xlim([xmin, xmax])
    ax_r.set_ylim([ymin, ymax])
    ax_r.set_aspect("equal", adjustable="box")
    ax_r.set_xticks([])
    ax_r.set_yticks([])
    ax_r.axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close(fig)

    logger.info(f"Saved CO2 pollution map to: {out_path}")


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / data_dir_name

    baseline_path = data_dir / baseline_filename
    v2x_path = data_dir / v2x_filename
    rl_path = data_dir / rl_filename

    sumo_cfg_path = project_root / "config" / "simulation.sumocfg"
    if not sumo_cfg_path.exists():
        raise FileNotFoundError(f"Expected config file not found: {sumo_cfg_path}")

    print(f"Using SUMO config: {sumo_cfg_path}")
    print(f"Reading parquet files from: {data_dir}")

    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline parquet.")

    if v2x_path.exists() and rl_path.exists():
        raise FileNotFoundError(f"Both V2X and RL files exists.\nPlease make sure to have just one")

    if not v2x_path.exists() and not rl_path.exists():
        raise FileNotFoundError(f"No V2X or RL parquet file found")

    comparison_path = v2x_path if v2x_path.exists() else rl_path
    comparison_label = "V2X" if v2x_path.exists() else "RL"

    df_base = pd.read_parquet(baseline_path)
    df_v2x = pd.read_parquet(comparison_path)

    net_path = _find_net_path_from_sumocfg(sumo_cfg_path)
    net = sumolib.net.readNet(str(net_path))
    edge_lengths = {e.getID(): e.getLength() for e in net.getEdges()}

    df_pollution = compute_edge_pollution_co2(df_base, df_v2x, edge_lengths)

    out_path = data_dir / f"co2_pollution_baseline_vs_{comparison_label}.png"
    plot_co2_pollution_map(
        df_pollution=df_pollution,
        sumo_config=sumo_cfg_path,
        out_path=out_path,
        title=f"Pollution map after {comparison_label}"
    )

    print(f"CO2 pollution geographic plot finished.")


if __name__ == "__main__":
    main()
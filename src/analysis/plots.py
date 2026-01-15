import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
import matplotlib as mpl

from pathlib import Path
from datacollector.data_collector import baseline_filename, v2x_filename, rule_based_filename, rl_filename, data_dir_name

RUN_LABELS = {"baseline": "Baseline", "rule_based": "Rule-based", "rl": "RL",}
RUN_ORDER = ["baseline", "rule_based", "rl"]
RUN_COLORS = {"baseline": "#0072B2", "rule_based": "#E69F00", "rl": "#009E73"}

def plot_filename(x: str, y: str, suffix: str = "png") -> str:
    def clean(name: str) -> str:
        return (
            name.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
        )
    return f"{clean(x)}_over_{clean(y)}.{suffix}"

def enforce_features(df, x: str, y: str):
    if x == y:
        raise ValueError(f"Plotting '{x}' against itself is not allowed.")
    for f in (x, y):
        if f not in df.columns:
            raise ValueError(f"Required feature '{f}' not found in dataset.")

def balanced_sample(df: pd.DataFrame, total_points: int, random_state: int = 42) -> pd.DataFrame:
    if "run" not in df.columns:
        raise ValueError("Column 'run' is missing; cannot stratify.")

    if len(df) <= total_points:
        return df

    counts = df["run"].value_counts()
    k = len(counts)
    smallest = counts.min()

    per_group = min(smallest, max(1, total_points // k))
    df_s = df.groupby("run", group_keys=False).sample(n=per_group, random_state=random_state)
    return df_s

def load_available_runs(data_dir: Path) -> dict[str, pd.DataFrame]:
    baseline_path = data_dir / baseline_filename
    rule_path = data_dir / rule_based_filename
    rl_path = data_dir / rl_filename
    legacy_v2x_path = data_dir / v2x_filename

    if not baseline_path.exists():
        raise FileNotFoundError(f"No baseline parquet found: {baseline_path}")

    runs: dict[str, pd.DataFrame] = {}

    df_base = pd.read_parquet(baseline_path)
    df_base["run"] = "baseline"
    runs["baseline"] = df_base

    if rule_path.exists():
        df_rule = pd.read_parquet(rule_path)
        df_rule["run"] = "rule_based"
        runs["rule_based"] = df_rule
    elif legacy_v2x_path.exists():
        df_rule = pd.read_parquet(legacy_v2x_path)
        df_rule["run"] = "rule_based"
        runs["rule_based"] = df_rule

    if rl_path.exists():
        df_rl = pd.read_parquet(rl_path)
        df_rl["run"] = "rl"
        runs["rl"] = df_rl

    if len(runs) < 2:
        raise FileNotFoundError(
            f"Need at least 2 runs to compare. Found: {list(runs.keys())} in {data_dir}"
        )

    return runs

def common_time_filter(runs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    max_times = [df["time"].max() for df in runs.values() if "time" in df.columns and not df.empty]
    time_common = min(max_times)

    filtered = {}
    for k, df in runs.items():
        if "time" in df.columns:
            filtered[k] = df[df["time"] <= time_common].copy()
        else:
            filtered[k] = df.copy()
    return filtered

def percent_improvement_lower_is_better(baseline: float, value: float) -> float:
    # positive = improvement
    if baseline is None or baseline == 0:
        return 0.0
    return 100.0 * (baseline - value) / baseline

def plot_co2_band_vs_feature(df: pd.DataFrame, out_dir: Path, feature: str,
                             bins: int = 60, min_bin_n: int = 200):
    if feature not in df.columns or "co2" not in df.columns or "run" not in df.columns:
        raise ValueError(f"Missing required columns for band plot: {feature}, co2, run")

    d = df[df["co2"] > 1].copy()
    if d.empty:
        return

    # Robust x-range shared across runs
    x_min, x_max = d[feature].quantile([0.01, 0.99])
    edges = np.linspace(float(x_min), float(x_max), bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0

    fig = plt.figure(figsize=(10.5, 6))

    for run_key in RUN_ORDER:
        sub = d[d["run"] == run_key]
        if sub.empty:
            continue

        x = sub[feature].to_numpy()
        y = sub["co2"].to_numpy()

        p10 = np.full(bins, np.nan)
        p50 = np.full(bins, np.nan)
        p90 = np.full(bins, np.nan)

        for i in range(bins):
            m = (x >= edges[i]) & (x < edges[i + 1])
            if m.sum() >= min_bin_n:
                yy = y[m]
                p10[i] = np.quantile(yy, 0.10)
                p50[i] = np.quantile(yy, 0.50)
                p90[i] = np.quantile(yy, 0.90)

        color = RUN_COLORS.get(run_key, None)
        label = RUN_LABELS.get(run_key, run_key)

        # Band + median line
        plt.fill_between(centers, p10, p90, alpha=0.18, color=color)
        plt.plot(centers, p50, linewidth=2.6, color=color, label=label)

    plt.title(f"CO2 vs {feature} (median and variability)")
    plt.xlabel(feature)
    plt.ylabel("CO2 emissions (mg/s)")
    plt.legend(frameon=True)
    plt.grid(True, alpha=0.25)

    medians = d.groupby("run")["co2"].median().to_dict()

    footer_parts = []
    if "baseline" in medians:
        base = medians["baseline"]

        if "rule_based" in medians:
            pct = 100.0 * (base - medians["rule_based"]) / base
            footer_parts.append(f"Rule-based vs Baseline: {pct:+.1f}%")

        if "rl" in medians:
            pct = 100.0 * (base - medians["rl"]) / base
            footer_parts.append(f"RL vs Baseline: {pct:+.1f}%")

    if footer_parts:
        fig.text(
            0.5, 0.01,
            " | ".join(footer_parts),
            ha="center",
            va="bottom",
            fontsize=10
        )
        plt.tight_layout(rect=[0, 0.05, 1, 1])
    else:
        plt.tight_layout()

    plt.savefig(out_dir / f"{feature}_co2_band.png", dpi=170)
    plt.close()



def plot_queue_time_over_time(df: pd.DataFrame, out_dir: Path):
    required = {"time", "queue_time", "run"}
    if not required.issubset(df.columns):
        print("Missing required columns for queue-time plot.")
        return

    grouped = df.groupby(["run", "time"])["queue_time"].mean().reset_index()

    with plt.style.context("seaborn-v0_8-darkgrid"):
        fig = plt.figure(figsize=(12, 6))
        final_means = {}

        for run, g in grouped.groupby("run"):
            label = RUN_LABELS.get(run, run)
            plt.plot(g["time"], g["queue_time"], label=label, linewidth=2)
            final_means[run] = g["queue_time"].iloc[-1]

        base_q = final_means.get("baseline")
        summary_lines = []
        if base_q is not None:
            for run in ("rule_based", "rl"):
                if run in final_means:
                    delta = final_means[run] - base_q
                    summary_lines.append(f"{RUN_LABELS[run]} vs Baseline: {delta:.0f}s (end-of-sim avg)")

        plt.xlabel("Simulation Time (s)")
        plt.ylabel("Average Queue Time per Vehicle (s)")
        plt.title("Queueing Time Over Time")
        plt.legend()

        if summary_lines:
            fig.text(0.5, 0.01, " | ".join(summary_lines), ha="center", fontsize=10, wrap=True)

        plt.ticklabel_format(style="plain", axis="y")
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(out_dir / plot_filename("queue_time", "time"), dpi=150)
        plt.close()

def plot_timeloss_histogram(df: pd.DataFrame, out_dir: Path):
    if "time_loss" not in df.columns or "run" not in df.columns:
        print("Missing 'time_loss' or 'run' column.")
        return

    with plt.style.context("seaborn-v0_8-darkgrid"):
        fig = plt.figure(figsize=(10, 6))

        medians = {}
        for run in sorted(df["run"].unique()):
            subset = df[df["run"] == run]["time_loss"].clip(upper=300)
            medians[run] = subset.median()
            plt.hist(subset, bins=30, alpha=0.55, label=RUN_LABELS.get(run, run))

        base_m = medians.get("baseline")
        summary_lines = []
        if base_m is not None:
            for run in ("rule_based", "rl"):
                if run in medians:
                    delta = base_m - medians[run]
                    summary_lines.append(f"{RUN_LABELS[run]} median improvement: {delta:.0f}s")

        plt.xlabel("Time Loss (s)")
        plt.ylabel("Vehicle Count")
        plt.title("Distribution of Vehicle Time Loss")
        plt.legend()

        if summary_lines:
            fig.text(0.5, 0.01, " | ".join(summary_lines), ha="center", fontsize=10, wrap=True)

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(out_dir / "timeloss_histogram.png", dpi=150)
        plt.close()

def plot_total_co2_pie(df: pd.DataFrame, out_dir: Path):
    if "co2" not in df.columns or "run" not in df.columns:
        print("Missing columns for CO2 pie chart.")
        return

    totals = df.groupby("run")["co2"].sum().sort_index()
    labels = [RUN_LABELS.get(r, r) for r in totals.index]
    values = totals.values

    with plt.style.context("seaborn-v0_8"):
        plt.figure(figsize=(6, 6))
        plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        plt.title("Total CO2 Emissions Share")
        plt.tight_layout()
        plt.savefig(out_dir / "co2_total_pie.png", dpi=150)
        plt.close()

def main(max_points: int):
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / data_dir_name
    out_dir = data_dir

    runs = load_available_runs(data_dir)
    runs = common_time_filter(runs)

    df_all = pd.concat(list(runs.values()), ignore_index=True)

    df_sampled = balanced_sample(df_all, total_points=max_points)

    plot_co2_band_vs_feature(df_all, out_dir, "accel")
    plot_co2_band_vs_feature(df_all, out_dir, "speed")
    plot_co2_band_vs_feature(df_all, out_dir, "jerk")
    plot_queue_time_over_time(df_sampled, out_dir)
    plot_timeloss_histogram(df_sampled, out_dir)
    plot_total_co2_pie(df_sampled, out_dir)

    print(f"All plots done. Runs included: {sorted(df_all['run'].unique())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run plots with custom sampling")
    parser.add_argument("--max-points", type=int, default=200000000)
    args = parser.parse_args()
    main(args.max_points)

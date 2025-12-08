import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from pathlib import Path
from data_collector import baseline_filename, v2x_filename, data_dir_name

def enforce_features(df, x: str, y: str):
    if x == y:
        raise ValueError(f"Plotting '{x}' against itself is not allowed.")
    for f in (x, y):
        if f not in df.columns:
            raise ValueError(f"Required feature '{f}' not found in dataset.")


def set_total_points(df, total_points, random_state=42):
    """There are too many plots to be plotted in a short time so we set samples"""
    if len(df) >= total_points:
        print(f"Sampling from {len(df)} to {total_points} for plotting")
        # Proving distribution between baseline and v2x is equal
        if "run" in df.columns:
            print("Distribution before sampling:", df["run"].value_counts().to_dict())
            smallest_class = df["run"].value_counts().min()
            max_allowed = 2 * smallest_class
            if total_points > max_allowed:
                raise ValueError(
                    f"total points = {total_points} exceeds the maximum allowed {max_allowed}"
                    f"(=2 baseline class size {smallest_class})"
                )
            # Balanced stratification
            limit = min(smallest_class, total_points//2)
            df_sampled = df.groupby("run").sample(n=limit, random_state=random_state)

            print("Distribution after sampling:", df_sampled["run"].value_counts().to_dict())
            return df_sampled

        raise ValueError(f"Column run is missing. Please make sure run column exists")
    else:
        return df


def plot_accel_vs_co2(df, out_dir):
    """Side-by-side accel vs CO2 plot."""
    enforce_features(df, "accel", "co2")

    df_filtered = df[df["co2"] > 1]
    df_base = df_filtered[df_filtered["run"] == "baseline"]
    df_v2x = df_filtered[df_filtered["run"] == "v2x"]

    with sns.axes_style("darkgrid"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        ax1, ax2 = axes

        sns.scatterplot(data=df_base, x="accel", y="co2", ax=ax1, alpha=0.25)
        sns.regplot(data=df_base, x="accel", y="co2", ax=ax1, scatter=False, line_kws={"color": "darkorange"})
        ax1.set_title("Baseline")
        ax1.set_xlabel("Acceleration (m/s²)")
        ax1.set_ylabel("CO2 emissions (mg/s)")

        sns.scatterplot(data=df_v2x, x="accel", y="co2", ax=ax2, alpha=0.25, color="darkorange")
        sns.regplot(data=df_v2x, x="accel", y="co2", ax=ax2, scatter=False, line_kws={"color": "royalblue"})
        ax2.set_title("V2X")
        ax2.set_xlabel("Acceleration (m/s²)")

        xmin = df_filtered["accel"].min()
        xmax = df_filtered["accel"].max()
        ax1.set_xlim(xmin, xmax)
        ax2.set_xlim(xmin, xmax)

        avg_base = df_base["co2"].mean()
        avg_v2x = df_v2x["co2"].mean()
        improvement = (avg_base - avg_v2x) / avg_base * 100

        summary = f"V2X reduced average CO2 emissions by {improvement:.1f}% compared to baseline."
        fig.text(0.5, 0.015, summary, ha='center', fontsize=12)

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(out_dir / "acceleration_over_CO2.png", dpi=150)
        plt.close()

def plot_speed_vs_co2(df, out_dir):
    enforce_features(df, "speed", "co2")

    df_filtered = df[df["co2"] > 1]
    df_base = df_filtered[df_filtered["run"] == "baseline"]
    df_v2x = df_filtered[df_filtered["run"] == "v2x"]

    with sns.axes_style("darkgrid"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        ax1, ax2 = axes

        sns.scatterplot(data=df_base, x="speed", y="co2", ax=ax1, alpha=0.3)
        sns.regplot(data=df_base, x="speed", y="co2", ax=ax1, scatter=False, line_kws={"color": "darkorange"})
        ax1.set_title("Baseline")
        ax1.set_xlabel("Speed (m/s)")
        ax1.set_ylabel("CO2 emissions (mg/s)")

        sns.scatterplot(data=df_v2x, x="speed", y="co2", ax=ax2, alpha=0.3, color="darkorange")
        sns.regplot(data=df_v2x, x="speed", y="co2", ax=ax2, scatter=False, line_kws={"color": "royalblue"})
        ax2.set_title("V2X")
        ax2.set_xlabel("Speed (m/s)")

        xmin = df_filtered["speed"].min()
        xmax = df_filtered["speed"].max()
        ax1.set_xlim(xmin, xmax)
        ax2.set_xlim(xmin, xmax)

        avg_base = df_base["co2"].mean()
        avg_v2x = df_v2x["co2"].mean()
        improvement = (avg_base - avg_v2x) / avg_base * 100

        summary = f"V2X reduced CO2 emissions by {improvement:.1f}% at comparable speeds."
        fig.text(0.5, 0.015, summary, ha='center', fontsize=12)

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(out_dir / "speed_over_CO2.png", dpi=150)
        plt.close()

def plot_co2_vs_jerk(df, out_dir):
    enforce_features(df, "jerk", "co2")

    df_filtered = df[df["co2"] > 1]
    df_base = df_filtered[df_filtered["run"] == "baseline"]
    df_v2x = df_filtered[df_filtered["run"] == "v2x"]

    with sns.axes_style("darkgrid"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        ax1, ax2 = axes

        sns.scatterplot(data=df_base, x="jerk", y="co2", ax=ax1, alpha=0.3)
        sns.regplot(data=df_base, x="jerk", y="co2", ax=ax1, scatter=False,line_kws={"color": "darkorange"})
        ax1.set_title("Baseline")
        ax1.set_xlabel("Jerk (m/s³)")
        ax1.set_ylabel("CO2 emissions (mg/s)")

        sns.scatterplot(data=df_v2x, x="jerk", y="co2", ax=ax2, alpha=0.3, color="orange")
        sns.regplot(data=df_v2x, x="jerk", y="co2", ax=ax2, scatter=False)
        ax2.set_title("V2X")
        ax2.set_xlabel("Jerk (m/s³)")

        xmin = df_filtered["jerk"].min()
        xmax = df_filtered["jerk"].max()
        ax1.set_xlim(xmin, xmax)
        ax2.set_xlim(xmin, xmax)

        var_base = df_base["jerk"].var()
        var_v2x = df_v2x["jerk"].var()
        improvement = (var_base - var_v2x) / var_base * 100

        summary = f"V2X improved driving smoothness by {improvement:.1f}%, reducing jerk variability."
        fig.text(0.5, 0.015, summary, ha='center', fontsize=12)

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(out_dir / "jerk_over_CO2.png", dpi=150)
        plt.close()

def compute_stop_durations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-vehicle stop durations and average stop time.

    We treat a vehicle as stopped if its speed is below 0.1 m/s.
    """
    df = df.copy()
    df["is_stopped"] = (df["speed"] < 0.1).astype(bool)
    labels = []

    for vid, veh_grp in df.groupby("veh_id"):
        tot_stop = veh_grp["is_stopped"].sum()
        no_stops = veh_grp["stops"].max()

        if no_stops > 0:
            avg_st_time = tot_stop / no_stops
        else:
            avg_st_time = 0.0

        labels.append({
            "veh_id": vid,
            "total_st_time": tot_stop,
            "avg_st_time": avg_st_time,
        })

    return pd.DataFrame(labels)


def plot_stop_duration_vs_speed(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Plot stop duration vs average speed.

    - If df has no 'run' column → single plot.
    - If df has 'baseline' and 'v2x' in 'run' → side-by-side comparison.
    """

    if not {"veh_id", "speed", "stops"}.issubset(df.columns):
        print("Missing 'veh_id', 'speed' or 'stops' columns for stop-duration plot.")
        return

    #Side-by-side baseline vs v2x
    df_base = df[df["run"] == "baseline"]
    df_v2x = df[df["run"] == "v2x"]

    if df_base.empty or df_v2x.empty:
        print("One of the runs (baseline/v2x) is empty; falling back to single plot.")
        return plot_stop_duration_vs_speed(df.drop(columns=["run"]), out_dir)

    def _prepare_stop_df(df_run: pd.DataFrame) -> pd.DataFrame:
        stop_df = compute_stop_durations(df_run)
        avg_speed = df_run.groupby("veh_id")["speed"].mean().reset_index(name="avg_speed")
        merged = pd.merge(stop_df, avg_speed, on="veh_id")
        return merged[
            (merged["avg_speed"] <= 20) &
            (merged["total_st_time"] <= 300)
        ]

    base_data = _prepare_stop_df(df_base)
    v2x_data = _prepare_stop_df(df_v2x)

    all_avg_speeds = pd.concat([base_data["avg_speed"], v2x_data["avg_speed"]])
    all_total_st = pd.concat([base_data["total_st_time"], v2x_data["total_st_time"]])

    x_min, x_max = all_avg_speeds.min(), all_avg_speeds.max()
    y_min, y_max = all_total_st.min(), all_total_st.max()

    with sns.axes_style("darkgrid"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        ax1, ax2 = axes

        # Baseline
        sns.scatterplot(
            data=base_data,
            x="avg_speed",
            y="total_st_time",
            color="#4682B4",
            alpha=0.5,
            s=30,
            ax=ax1,
        )
        sns.regplot(
            data=base_data,
            x="avg_speed",
            y="total_st_time",
            scatter=False,
            color="#FF6347",
            line_kws={"lw": 2},
            ax=ax1,
        )
        ax1.set_title("Baseline", style="italic")
        ax1.set_xlabel("Average Speed (m/s)")
        ax1.set_ylabel("Total Time Spent Below 0.1 m/s")

        sns.scatterplot(
            data=v2x_data,
            x="avg_speed",
            y="total_st_time",
            color="#FFA500",
            alpha=0.5,
            s=30,
            ax=ax2,
        )
        sns.regplot(
            data=v2x_data,
            x="avg_speed",
            y="total_st_time",
            scatter=False,
            color="#FF4500",
            line_kws={"lw": 2},
            ax=ax2,
        )
        ax2.set_title("V2X", style="italic")
        ax2.set_xlabel("Average Speed (m/s)")
        ax2.set_ylabel("")

        ax1.set_xlim(x_min, x_max)
        ax2.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax2.set_ylim(y_min, y_max)

        mean_base = base_data["total_st_time"].mean()
        mean_v2x = v2x_data["total_st_time"].mean()
        improvement = (mean_base - mean_v2x) / mean_base * 100

        summary = f"V2X reduced total stop time by {improvement:.1f}% compared to baseline."
        fig.text(0.5, 0.015, summary, ha='center', fontsize=12)

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(out_dir / "stop_duration_over_speed.png", dpi=150)
        plt.close()

def main(max_points):
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / data_dir_name

    baseline_path = data_dir / baseline_filename
    v2x_path = data_dir / v2x_filename

    if not baseline_path.exists():
        print(f"No {baseline_path} file found.")
        return

    if not v2x_path.exists():
        print(f"Missing V2X file: {v2x_path}")
        return

    out_dir = data_dir

    df_v2x = pd.read_parquet(v2x_path)
    df_v2x["run"] = "v2x"

    if baseline_path.exists():
        df_baseline = pd.read_parquet(baseline_path)
        df_baseline["run"] = "baseline"
        df = pd.concat([df_baseline, df_v2x], ignore_index=True)
        print(f"Using dataset: {baseline_path}\nv2x at {v2x_path}\n"
              f"Generating comparison plots (baseline vs v2x)")
    else:
        df = df_v2x
        print(f"Using datasets: {v2x_path}\n"
              f"(No {baseline_path.name} found, plotting single run.)")

    df_sampled = set_total_points(df, total_points=max_points)

    plot_accel_vs_co2(df_sampled, out_dir)
    plot_speed_vs_co2(df_sampled, out_dir)
    plot_co2_vs_jerk(df_sampled, out_dir)
    plot_stop_duration_vs_speed(df_sampled, out_dir)

    print("All plots are done")

if __name__ == "__main__":
    #Initialize parser for no. of sampled points
    parser = argparse.ArgumentParser(description="Run plots with custom sampling")
    parser.add_argument("--max-points",
                        type=int,
                        default=2000000,
                        help="Maximum number of sampled points used in plotting"
)
    args = parser.parse_args()
    main(args.max_points)

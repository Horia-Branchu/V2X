import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def find_latest_csv(root_dir: Path, filename="vehicles.csv"):
    """Find vehicles.csv file within the project"""

    candidates = list(root_dir.rglob(filename))
    if len(candidates) == 0:
        return None
    #Sorting files by the last modification time
    candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return candidates[0]


def plot_accel_vs_co2(df, out_dir):
    """Side-by-side accel vs CO2 plot."""

    if not {"accel", "co2"}.issubset(df.columns):
        return

    df_filtered = df[df["co2"] > 1]

    if "run" not in df.columns:
        plt.figure(figsize=(6, 6))
        sns.regplot(data=df_filtered, x="accel", y="co2")
        plt.title("Acceleration vs CO2")
        plt.savefig(out_dir / "plot_accel_co2.png", dpi=150)
        plt.close()
        return

    df_base = df_filtered[df_filtered["run"] == "baseline"]
    df_v2x = df_filtered[df_filtered["run"] == "v2x"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    ax1, ax2 = axes

    sns.scatterplot(data=df_base, x="accel", y="co2", ax=ax1, alpha=0.25)
    sns.regplot(data=df_base, x="accel", y="co2", ax=ax1, scatter=False)
    ax1.set_title("Acceleration vs CO2 (Baseline)")

    sns.scatterplot(data=df_v2x, x="accel", y="co2", ax=ax2, alpha=0.25, color="darkorange")
    sns.regplot(data=df_v2x, x="accel", y="co2", ax=ax2, scatter=False, line_kws={"color": "darkred"})
    ax2.set_title("Acceleration vs CO2 (V2X)")

    plt.tight_layout()
    plt.savefig(out_dir / "plot_accel_co2_side_by_side.png", dpi=150)
    plt.close()

def plot_speed_vs_co2(df, out_dir):
    if not {"speed", "co2"}.issubset(df.columns):
        return

    if "run" not in df.columns:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x="speed", y="co2", alpha=0.3)
        plt.title("CO2 vs Speed")
        plt.savefig(out_dir / "plot_co2_speed.png", dpi=150)
        plt.close()
        return

    df_base = df[df["run"] == "baseline"]
    df_v2x = df[df["run"] == "v2x"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    ax1, ax2 = axes

    sns.scatterplot(data=df_base, x="speed", y="co2", ax=ax1, alpha=0.3)
    ax1.set_title("CO2 vs Speed (Baseline)")

    sns.scatterplot(data=df_v2x, x="speed", y="co2", ax=ax2, alpha=0.3, color="orange")
    ax2.set_title("CO2 vs Speed (V2X)")

    plt.tight_layout()
    plt.savefig(out_dir / "plot_speed_co2_side_by_side.png", dpi=150)
    plt.close()

def plot_co2_vs_jerk(df, out_dir):
    if not {"jerk", "co2"}.issubset(df.columns):
        return

    df_filtered = df[(df["jerk"] < 3.3) & (df["co2"] >= 2000)]

    if "run" not in df_filtered.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_filtered, x="jerk", y="co2", alpha=0.4)
        plt.title("CO2 vs Jerk")
        plt.savefig(out_dir / "plot_co2_jerk.png", dpi=150)
        plt.close()
        return

    df_base = df_filtered[df_filtered["run"] == "baseline"]
    df_v2x = df_filtered[df_filtered["run"] == "v2x"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    ax1, ax2 = axes

    sns.scatterplot(data=df_base, x="jerk", y="co2", ax=ax1, alpha=0.3)
    sns.regplot(data=df_base, x="jerk", y="co2", ax=ax1, scatter=False)
    ax1.set_title("CO2 vs Jerk (Baseline)")

    sns.scatterplot(data=df_v2x, x="jerk", y="co2", ax=ax2, alpha=0.3, color="orange")
    sns.regplot(data=df_v2x, x="jerk", y="co2", ax=ax2, scatter=False)
    ax2.set_title("CO2 vs Jerk (V2X)")

    plt.tight_layout()
    plt.savefig(out_dir / "plot_co2_jerk_side_by_side.png", dpi=150)
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

    # ---------- Single-run behaviour ----------
    if "run" not in df.columns:
        stop_df = compute_stop_durations(df)
        avg_speed = df.groupby("veh_id")["speed"].mean().reset_index(name="avg_speed")

        merged = pd.merge(stop_df, avg_speed, on="veh_id")
        merged = merged[
            (merged["avg_speed"] <= 20) &
            (merged["total_st_time"] <= 300)
        ]

        with sns.axes_style("darkgrid"):
            plt.figure(figsize=(8, 6))
            sns.scatterplot(
                data=merged,
                x="avg_speed",
                y="total_st_time",
                color="#4682B4",
                alpha=0.5,
                s=30,
            )
            sns.regplot(
                data=merged,
                x="avg_speed",
                y="total_st_time",
                scatter=False,
                color="#FF6347",
                ci=90,
                line_kws={"lw": 2},
            )
            plt.xlabel("Average Speed (m/s)")
            plt.ylabel("Total Time Spent Below 0.1 m/s")
            plt.title("Stop Duration vs Average Speed", style="italic")
            plt.tight_layout()
            plt.savefig(out_dir / "plot_stop_duration_vs_speed.png", dpi=150)
            plt.close()
        return

    # ---------- Side-by-side baseline vs v2x ----------
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
            ci=90,
            line_kws={"lw": 2},
            ax=ax1,
        )
        ax1.set_title("Stop Duration vs Avg Speed (Baseline)", style="italic")
        ax1.set_xlabel("Average Speed (m/s)")
        ax1.set_ylabel("Total Time Spent Below 0.1 m/s")

        # V2X
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
            ci=90,
            line_kws={"lw": 2},
            ax=ax2,
        )
        ax2.set_title("Stop Duration vs Avg Speed (V2X)", style="italic")
        ax2.set_xlabel("Average Speed (m/s)")
        ax2.set_ylabel("")

        ax1.set_xlim(x_min, x_max)
        ax2.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax2.set_ylim(y_min, y_max)

        plt.tight_layout()
        plt.savefig(out_dir / "plot_stop_duration_vs_speed_side_by_side.png", dpi=150)
        plt.close()

def main():
    root = Path(__file__).resolve().parents[1]
    csv_path = find_latest_csv(root, filename="vehicles.csv")
    if csv_path is None:
        print("No vehicles.csv file found.")
        return

    out_dir = csv_path.parent

    baseline_path = out_dir / "vehicles_baseline.csv"

    df_v2x = pd.read_csv(csv_path, low_memory=False)
    df_v2x["run"] = "v2x"

    if baseline_path.exists():
        df_baseline = pd.read_csv(baseline_path, low_memory=False)
        df_baseline["run"] = "baseline"
        df = pd.concat([df_baseline, df_v2x], ignore_index=True)
        print(df["run"].value_counts())
        print(f"Using dataset: {csv_path} + baseline at {baseline_path}\n"
              f"Generating comparison plots (baseline vs v2x)...")
    else:
        df = df_v2x
        print(f"Using dataset: {csv_path}\n"
              f"(No vehicles_baseline.csv found, plotting single run.)")
    plot_accel_vs_co2(df, out_dir)
    plot_speed_vs_co2(df, out_dir)
    plot_co2_vs_jerk(df, out_dir)
    plot_stop_duration_vs_speed(df, out_dir)

    print("All plots are done")

if __name__ == "__main__":
    main()

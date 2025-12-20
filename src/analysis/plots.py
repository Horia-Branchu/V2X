import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from pathlib import Path
from data_collector import baseline_filename, v2x_filename, data_dir_name, rl_filename

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


def plot_accel_vs_co2(df, out_dir, title):
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
        ax2.set_title(title)
        ax2.set_xlabel("Acceleration (m/s²)")

        xmin = df_filtered["accel"].min()
        xmax = df_filtered["accel"].max()
        ax1.set_xlim(xmin, xmax)
        ax2.set_xlim(xmin, xmax)

        avg_base = df_base["co2"].mean()
        avg_v2x = df_v2x["co2"].mean()
        improvement = (avg_base - avg_v2x) / avg_base * 100

        summary = f"{title} reduced average CO2 emissions by {improvement:.1f}% compared to baseline."
        fig.text(0.5, 0.015, summary, ha='center', fontsize=12)

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(out_dir / plot_filename("accel", "co2"), dpi=150)
        plt.close()

def plot_speed_vs_co2(df, out_dir, title):
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
        ax2.set_title(title)
        ax2.set_xlabel("Speed (m/s)")

        xmin = df_filtered["speed"].min()
        xmax = df_filtered["speed"].max()
        ax1.set_xlim(xmin, xmax)
        ax2.set_xlim(xmin, xmax)

        avg_base = df_base["co2"].mean()
        avg_v2x = df_v2x["co2"].mean()
        improvement = (avg_base - avg_v2x) / avg_base * 100

        summary = f"{title} reduced CO2 emissions by {improvement:.1f}% at comparable speeds."
        fig.text(0.5, 0.015, summary, ha='center', fontsize=12)

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(out_dir / plot_filename("speed", "co2"), dpi=150)
        plt.close()

def plot_co2_vs_jerk(df, out_dir, title):
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
        ax2.set_title(title)
        ax2.set_xlabel("Jerk (m/s³)")

        xmin = df_filtered["jerk"].min()
        xmax = df_filtered["jerk"].max()
        ax1.set_xlim(xmin, xmax)
        ax2.set_xlim(xmin, xmax)

        var_base = df_base["jerk"].var()
        var_v2x = df_v2x["jerk"].var()
        improvement = (var_base - var_v2x) / var_base * 100

        summary = f"{title} improved driving smoothness by {improvement:.1f}%, reducing jerk variability."
        fig.text(0.5, 0.015, summary, ha='center', fontsize=12)

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(out_dir / plot_filename("jerk", "co2"), dpi=150)
        plt.close()


def plot_queue_time_over_time(df: pd.DataFrame, out_dir: Path, title:str):
    required = {"time", "queue_time", "run"}
    if not required.issubset(df.columns):
        print("Missing required columns for queue-time plot.")
        return

    grouped = (
        df.groupby(["run", "time"])["queue_time"]
        .mean()
        .reset_index()
    )

    with plt.style.context("seaborn-v0_8-darkgrid"):
        fig = plt.figure(figsize=(12, 6))

        final_means = {}

        for run, g in grouped.groupby("run"):
            if run == "baseline":
                label = "Baseline"
            else:
                label = "RL"

            plt.plot(
                g["time"],
                g["queue_time"],
                label=label,
                linewidth=2
            )
            final_means[run] = g["queue_time"].iloc[-1]

        base_q = final_means["baseline"]
        v2x_q = final_means["v2x"]
        delta_sec = v2x_q - base_q

        plt.xlabel("Simulation Time (s)")
        plt.ylabel("Average Queue Time per Vehicle (s)")
        plt.title("Queueing Time Evolution Over Time")
        plt.legend()

        summary = (
            f"By the end of the simulation, {title} vehicles accumulated on average "
            f"{delta_sec:.0f} additional seconds of queueing time. "
            f"This reflects intentional holding of some vehicles to reduce stop-and-go traffic "
            f"and improve overall flow stability."
        )

        fig.text(0.5, 0.01, summary, ha="center", fontsize=10, wrap=True)

        plt.ticklabel_format(style="plain", axis="y")
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(out_dir / plot_filename("queue_time", "time"), dpi=150)
        plt.close()



def plot_timeloss_histogram(df: pd.DataFrame, out_dir: Path, title:str):
    if "time_loss" not in df.columns or "run" not in df.columns:
        print("Missing 'time_loss' or 'run' column.")
        return

    with plt.style.context("seaborn-v0_8-darkgrid"):
        fig = plt.figure(figsize=(10, 6))

        medians = {}

        for run, color in [("baseline", "steelblue"), ("v2x", "darkorange")]:
            subset = df[df["run"] == run]["time_loss"]
            medians[run] = subset.median()

            if run == "baseline":
                label = "Baseline"
            else:
                label = "RL"

            plt.hist(
                subset.clip(upper=300),
                bins=30,
                alpha=0.6,
                label=label,
            )

        delta_sec = medians["baseline"] - medians["v2x"]

        plt.xlabel("Time Loss (s)")
        plt.ylabel("Vehicle Count")
        plt.title("Distribution of Vehicle Time Loss")
        plt.legend()

        summary = (
            f"{title} reduced median vehicle time loss by {delta_sec:.0f} seconds compared to baseline. "
            f"High-delay cases are concentrated among fewer vehicles rather than distributed "
            f"across the entire traffic stream."
        )

        fig.text(0.5, 0.01, summary, ha="center", fontsize=10, wrap=True)

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(out_dir / "timeloss_histogram.png", dpi=150)
        plt.close()



def plot_total_co2_pie(df: pd.DataFrame, out_dir: Path, title:str):
    if "co2" not in df.columns or "run" not in df.columns:
        print("Missing columns for CO2 pie chart.")
        return

    totals = df.groupby("run")["co2"].sum()

    labels = ["Baseline",title]
    values = totals.values

    with plt.style.context("seaborn-v0_8"):
        plt.figure(figsize=(6, 6))
        plt.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90
        )
        plt.title(f"Total CO2 Emissions: Baseline vs {title}")
        plt.tight_layout()
        plt.savefig(out_dir / "co2_total_pie.png", dpi=150)
        plt.close()


def main(max_points):
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / data_dir_name

    baseline_path = data_dir / baseline_filename
    v2x_path = data_dir / v2x_filename
    rl_path = data_dir / rl_filename

    if not baseline_path.exists():
        raise FileNotFoundError(f"No {baseline_path} file found.")

    if v2x_path.exists() and rl_path.exists():
        raise FileNotFoundError(f"Both V2X and RL files exists.\nPlease make sure to have just one")

    if not v2x_path.exists() and not rl_path.exists():
        raise FileNotFoundError(f"No V2X or RL parquet file found")

    if v2x_path.exists():
        comparison_path = v2x_path
        title = "V2X"
    else:
        comparison_path = rl_path
        title = "RL"

    out_dir = data_dir

    df_v2x = pd.read_parquet(comparison_path)
    df_v2x["run"] = "v2x"

    if baseline_path.exists():
        df_baseline = pd.read_parquet(baseline_path)
        df_baseline["run"] = "baseline"
        df = pd.concat([df_baseline, df_v2x], ignore_index=True)
        #Limited analysis to the common simulation duration for a fair comparison
        time_baseline_end = df_baseline["time"].max()
        time_v2x_end = df_v2x["time"].max()
        time_common = min(time_baseline_end, time_v2x_end)

        df = df[df["time"] <= time_common]
        print(f"Using dataset: {baseline_path}\nv2x at {comparison_path}\n"
              f"Generating comparison plots (baseline vs v2x)")
    else:
        df = df_v2x
        print(f"Using datasets: {comparison_path}\n"
              f"(No {baseline_path.name} found, plotting single run.)")

    df_sampled = set_total_points(df, total_points=max_points)

    plot_accel_vs_co2(df_sampled, out_dir, title)
    plot_speed_vs_co2(df_sampled, out_dir, title)
    plot_co2_vs_jerk(df_sampled, out_dir, title)
    plot_queue_time_over_time(df_sampled, out_dir, title)
    plot_timeloss_histogram(df_sampled, out_dir, title)
    plot_total_co2_pie(df_sampled, out_dir, title)

    print("All plots are done")

if __name__ == "__main__":
    #Initialize parser for no. of sampled points
    parser = argparse.ArgumentParser(description="Run plots with custom sampling")
    parser.add_argument("--max-points",
                        type=int,
                        default=200000,
                        help="Maximum number of sampled points used in plotting"
)
    args = parser.parse_args()
    main(args.max_points)
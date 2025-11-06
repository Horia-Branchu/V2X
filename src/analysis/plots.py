import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def find_latest_csv(root_dir: Path, filename="vehicles.csv"):
    """Find vehicles.csv file within the project"""

    candidates = list(root_dir.rglob(filename))
    if len(candidates) == 0:
        return None
    ###Sorting files by the last modification time
    candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return candidates[0]

def plot_co2_over_time(df, out_dir):
    """Plot average CO2 emission over simulation time"""

    if not {"time", "co2"}.issubset(df.columns):
        return None

    #I limited t = 860 because vehicles start disappearing afterwards
    df_until_900 = df[df["time"] <= 860]

    avg_by_time = df_until_900.groupby("time")["co2"].mean()
    with sns.axes_style("darkgrid"):
        plt.figure(figsize=(8, 5))
        plt.plot(avg_by_time.index, avg_by_time.values, color="#3C096C")
        plt.xlabel("Time (s)")
        plt.ylabel("Average CO2")
        plt.title("Average Network CO2 Emission Over Time", style='italic')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(out_dir / "plot_co2_time.png", dpi=150)
        plt.close()

def plot_accel_vs_co2(df, out_dir):
    """Plot acceleration versus CO2 emission with regression line"""

    if not {"accel", "co2"}.issubset(df.columns):
        return None

    df_filtered = df[df["co2"] > 1]
    with sns.axes_style("darkgrid"):
        plt.figure(figsize=(6, 6))
        sns.regplot(
            data=df_filtered,
            x="accel",
            y="co2",
            scatter_kws={'alpha': 0.2, 's': 10, 'color': '#3C096C'},
            line_kws={'color': '#FFC880', 'linewidth': 2}
        )
        plt.title("Acceleration vs CO2 Emission", style='italic')
        plt.xlabel("Acceleration (m/s)")
        plt.ylabel("CO2 (mg/s)")
        plt.tight_layout()
        plt.savefig(out_dir / "plot_accel_co2.png", dpi=150)
        plt.close()

def plot_speed_vs_co2(df, out_dir):
    """Plot speed versus CO2 emission with regression trendline"""

    if not {"speed", "co2"}.issubset(df.columns):
        return None
    df_non0 = df[(df["co2"] > 1) & (df["speed"] > 1)]
    with sns.axes_style("whitegrid"):
        plt.figure(figsize=(8, 6))
        sns.regplot(
            data=df_non0,
            x="speed",
            y="co2",
            scatter_kws={'alpha': 0.9, 's': 10, 'color': '#6C5952'},
            line_kws={'color': '#355E3B', 'linewidth': 2}
        )
        plt.title("Speed vs CO2 Emission", style='italic')
        plt.xlabel("Speed (m/s)")
        plt.ylabel("CO2 (mg/s)")
        plt.tight_layout()
        plt.savefig(out_dir / "plot_speed_co2.png", dpi=150)
        plt.close()


def plot_co2_vs_jerk(df, out_dir):
    """Plot CO2 emission versus Jerk"""

    if not {"jerk", "co2"}.issubset(df.columns):
        print(f"Missing jerk or co2 column for co2 vs jerk plot")
        return None

    with sns.axes_style("darkgrid"):
        plt.figure(figsize=(8, 6))
        df_filtered = df[(df["jerk"] < 3.3) & (df["co2"] >= 2000)] #df["co2"] >= 2000) => vechicle is moving
        sns.scatterplot(
            data=df_filtered,
            x="jerk",
            y="co2",
            color="royalblue",
            alpha=0.4,
            s=10,
            label="Data points"
        )
        sns.regplot(
            data=df_filtered,
            x="jerk",
            y="co2",
            scatter=False,
            line_kws={'color': '#FFD700', 'linewidth': 2},
        )
        #Just a preference in this case
        plt.legend().remove()
        plt.title("CO2 Emission vs Jerk", style='italic')
        plt.xlabel("Jerk")
        plt.ylabel("CO2 Emission(mg/s)")
        plt.tight_layout()
        plt.savefig(out_dir / "plot_co2_jerk.png", dpi=150)
        plt.close()


def plot_min_speed_per_edge(df, out_dir):
    """Plot top routes or edges by minimum vehicle speed
        We set it to 30"""
    if not {"edge", "speed"}.issubset(df.columns):
        print(f"Missing edge or speed column for max speed per edge")
        return None

    df_min_speed_per_edge = df.groupby('edge')['speed'].max().reset_index()
    top_smaller = df_min_speed_per_edge.sort_values(by='speed', ascending=True).head(30)


    plt.figure(figsize=(14, 7))
    sns.barplot(
        data=top_smaller,
        x='edge', y='speed',
        hue='edge',
        palette='viridis',
        legend=False
    )
    plt.title('Top 20 Minimum Speed per Edge', style='italic')
    plt.xlabel('Route/Edge')
    plt.ylabel('Minimum Speed (m/s)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_min_speed_per_edge.png", dpi=150)
    plt.close()


def plot_speed_over_time(df, out_dir):
    """Plot average vehicle speed over time"""

    if not {"time", "speed"}.issubset(df.columns):
        print(f"Missing 'time' or 'speed' column for speed over time.")
        return None

    with sns.axes_style("darkgrid"):
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='time', y='speed', color='royalblue')
        plt.title('Speed over Time')
        plt.xlabel('Time')
        plt.ylabel('Speed (m/s)')
        plt.tight_layout()
        plt.savefig(out_dir / "plot_speed_over_time.png", dpi=150)
        plt.close()

def main():
    """Find the latest vehicles.csv and generate all performance plots"""

    project_root = Path(__file__).resolve().parents[1]
    csv_path = find_latest_csv(project_root, "vehicles.csv")

    if not csv_path or not csv_path.exists():
        print(f"No vehicles.csv found. Run the simulation first.")
        return None

    #Timestamp for starting the dataset
    print(f"Using dataset: {csv_path}\n"
          f"Waiting for the plots")

    df = pd.read_csv(csv_path,low_memory=False)
    out_dir = csv_path.parent

    #Displaying Functions
    plot_co2_over_time(df, out_dir)
    plot_accel_vs_co2(df, out_dir)
    plot_speed_vs_co2(df, out_dir)
    plot_min_speed_per_edge(df, out_dir)
    plot_speed_over_time(df, out_dir)
    plot_co2_vs_jerk(df, out_dir)
    #Timestamp for dataset ending
    print(f"All plots are done")


if __name__ == "__main__":
    main()

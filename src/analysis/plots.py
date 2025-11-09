import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path



def find_latest_csv(root_dir: Path, filename="vehicles.csv"):
    """Find vehicles.csv file within the project"""

    candidates = list(root_dir.rglob(filename))
    if len(candidates) == 0:
        return None
    ###Sorting files by the last modification time
    candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return candidates[0]

def plot_speed_vs_route_avg_speed(df, out_dir):
    """Plot Speed vs Route Average Speed"""

    if not {"speed", "route_avg_speed"}.issubset(df.columns):
        print(f"Missing speed or route_avg_speed columns.")
        return None

    with sns.axes_style("darkgrid"):
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            data=df,
            x="route_avg_speed",
            y="speed",
            color="#3CB371",
            alpha=0.25,
            s=10
        )
        sns.regplot(
            data=df,
            x="route_avg_speed",
            y="speed",
            scatter=False,
            color="#2F4F4F",
            line_kws={"lw": 2}
        )
        plt.xlabel("Speed")
        plt.ylabel("Route Average Speed (m/s)")
        plt.title("Speed vs Route Average Speed (m/s) Before V2X", style="italic")
        plt.tight_layout()
        plt.savefig(out_dir / "plot_speed_vs_route_avg_speed.png", dpi=150)
        plt.close()

def plot_accel_vs_co2(df, out_dir):
    """Plot acceleration versus CO2 emission with regression line"""

    if not {"accel", "co2"}.issubset(df.columns):
        print(f"Missing accel or co2 column")
        return None

    df_filtered = df[df["co2"] > 1]
    with sns.axes_style("darkgrid"):
        plt.figure(figsize=(6, 6))
        sns.regplot(
            data=df_filtered,
            x="accel",
            y="co2",
            scatter_kws={'alpha': 0.4, 's': 10, 'color': '#3C096C'},
            line_kws={'color': '#FFC880', 'linewidth': 2}
        )
        plt.title("Acceleration vs CO2 Emission Before V2X", style='italic')
        plt.xlabel("Acceleration (m/s)")
        plt.ylabel("CO2 (mg/s)")
        plt.tight_layout()
        plt.savefig(out_dir / "plot_accel_co2.png", dpi=150)
        plt.close()
        return None

def plot_speed_vs_co2(df, out_dir):
    """Plot speed versus CO2 emission with regression trendline"""

    if not {"speed", "co2"}.issubset(df.columns):
        print("Missing 'speed' or 'co2' column")
        return None

    df_non0 = df[(df["co2"] > 1) & (df["speed"] > 1)]
    with sns.axes_style("whitegrid"):
        plt.figure(figsize=(8, 6))
        sns.regplot(
            data=df_non0,
            x="speed",
            y="co2",
            scatter_kws={'alpha': 0.6, 's': 10, 'color': '#1A80BB'},
            line_kws={'color': '#B8B8B8', 'linewidth': 2}
        )
        plt.title("Speed vs CO2 Emission Before V2X", style='italic')
        plt.xlabel("Speed (m/s)")
        plt.ylabel("CO2 (mg/s)")
        plt.tight_layout()
        plt.savefig(out_dir / "plot_speed_co2.png", dpi=150)
        plt.close()
        return None

def plot_co2_vs_jerk(df, out_dir):
    """Plot CO2 emission versus Jerk"""

    if not {"jerk", "co2"}.issubset(df.columns):
        print(f"Missing jerk or co2 column")
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
        plt.title("CO2 Emission vs Jerk Before V2X", style='italic')
        plt.xlabel("Jerk")
        plt.ylabel("CO2 Emission(mg/s)")
        plt.tight_layout()
        plt.savefig(out_dir / "plot_co2_jerk.png", dpi=150)
        plt.close()
        return None

def compute_stop_durations(df):
    """We treat a car as stopped if it s speed it s below 0.1
    (the same logic as in the data_collector.py file)"""

    df["is_stopped"] = (df["speed"] < 0.1).astype(bool)
    labels = []
    #Creating "tables" for each veh id with df.groupby"
    for vid, veh_grp in df.groupby("veh_id"):
        tot_stop = veh_grp["is_stopped"].sum()
        no_stops = veh_grp["stops"].max()
        if no_stops > 0:
            avg_st_time = tot_stop/no_stops
        else:
            #To avoid dividing it by 0
            avg_st_time = 0.1
        labels.append({
            "veh_id": vid,
            "total_st_time": tot_stop,
            "avg_st_time": avg_st_time
        })
    return pd.DataFrame(labels)

def plot_stop_duration_vs_speed(df, out_dir):
    stop_df = compute_stop_durations(df)
    #Grouping mean speed per veh id
    df_avg_speed = df.groupby("veh_id")["speed"].mean()
    #Resetting column direction
    df_avg_speed.reset_index(name="avg_speed")
    df_st_time = pd.merge(stop_df, df_avg_speed, on="veh_id")
    df_st_time = df_st_time[(df_st_time["avg_speed"] <= 20) & (df_st_time["total_st_time"] <= 300)]

    with sns.axes_style("darkgrid"):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df_st_time,
            x="avg_speed",
            y="total_st_time",
            color="#4682B4",  # steel blue
            alpha=0.5,
            s=30
        )
        sns.regplot(
            data=df_st_time,
            x="avg_speed",
            y="total_st_time",
            scatter=False,
            color="#FF6347",
            line_kws={"lw": 2}
        )
        plt.xlabel("Average Speed (m/s)")
        plt.ylabel("Total Time Spent Below 1m/s")
        plt.title("Stop Duration vs Average Speed Before V2X", style="italic")
        plt.tight_layout()
        plt.savefig(out_dir / "plot_stop_duration_vs_speed.png", dpi=150)
        plt.close()


def main():
    """Find the latest vehicles.csv and generate all performance plots"""

    project_root = Path(__file__).resolve().parents[1]
    csv_path = find_latest_csv(project_root, "vehicles.csv")

    if not csv_path or not csv_path.exists():
        print(f"No vehicles.csv found. Run the simulation first")
        return None

    #Timestamp for starting the dataset
    print(f"Using dataset: {csv_path}\n"
          f"Waiting for the plots")

    df = pd.read_csv(csv_path,low_memory=False)
    out_dir = csv_path.parent

    #Displaying Functions
    plot_speed_vs_route_avg_speed(df, out_dir)
    plot_accel_vs_co2(df, out_dir)
    plot_speed_vs_co2(df, out_dir)
    plot_co2_vs_jerk(df, out_dir)
    plot_stop_duration_vs_speed(df, out_dir)
    print(f"All plots are done")


if __name__ == "__main__":
    main()

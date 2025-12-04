import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from data_collector import vehicle_filename


def find_latest_csv(root_dir: Path, filename=vehicle_filename):
    """Search for the newest matching CSV in the project folder"""

    print(f"Correlation Map for {root_dir}\n"
          f"Correlation Map is generating")

    candidates = list(root_dir.rglob(filename))
    if len(candidates) == 0:
        return None

    ###Sorting files by the last modification time
    candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return candidates[0]

def main():
    """Generate a Pearson correlation heatmap without time and veh_id and save it"""

    project_root = Path(__file__).resolve().parents[2]
    csv_path = find_latest_csv(project_root, vehicle_filename)

    if not csv_path or not csv_path.exists():
        print(f"Could not find any {vehicle_filename} in the project"
              "Run the simulation first so DataCollector generates it")
        return None

    df = pd.read_csv(csv_path,low_memory=False)
    num_df = df.select_dtypes(include=np.number)
    if num_df.empty:
        print("No numeric columns found for correlation")
        return None

    columns_to_remove = ["veh_id", "trip_time", "distance"]
    for column in columns_to_remove:
        if column  in  num_df.columns:
            num_df.drop(column, axis=1, inplace=True)

    corr_matrix = num_df.corr(method="pearson")

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar_kws={"label": "Label Correlation"},
        square=True
    )
    plt.title("Feature Correlation Map with CO2 Consumption", fontsize=14, pad=12)
    plt.tight_layout()
    output_path = csv_path.parent / "co2_correlation_map.png"
    plt.savefig(output_path, dpi=150)
    print("Done")

if __name__ == "__main__":
    print(__file__)
    main()

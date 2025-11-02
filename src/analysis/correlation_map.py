#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def find_latest_csv(root_dir: Path, filename="vehicles.csv"):
    """Search for the newest matching CSV in the project folder"""

    candidates = list(root_dir.rglob(filename))
    if not candidates:
        return None
    candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return candidates[0]

def generate_correlation_map():
    """Generate a Pearson correlation heatmap without time and veh_id and save it"""

    project_root = Path(__file__).resolve().parents[1]
    csv_path = find_latest_csv(project_root, "vehicles.csv")

    if not csv_path or not csv_path.exists():
        print(f"Could not find any vehicles.csv file in the project"
              "Run the simulation first so DataCollector generates it")
        return


    df = pd.read_csv(csv_path,low_memory=False)
    #Dropping irrelevant columns
    cols_to_drop = [col for col in ["time", "veh_id"] if col in df.columns]
    if cols_to_drop:

        df_filtered = df.drop(columns=cols_to_drop)
    else:
        df_filtered = df.copy()

    # Saving filtered copy of the file
    filtered_path = csv_path.parent / "vehicles_filtered.csv"
    df_filtered.to_csv(filtered_path, index=False)

    num_df = df_filtered.select_dtypes(include=np.number)
    if num_df.empty:
        print("No numeric columns found for correlation")
        return

    corr_matrix = num_df.corr(method="pearson")
    # Sorting columns so CO2 appears first in the heatmap
    cols_sorted = sorted(corr_matrix.columns, key=lambda c: 0 if "co2" in c.lower() else 1)
    corr_matrix = corr_matrix.loc[cols_sorted, cols_sorted]

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar_kws={"label": "Pearson correlation"},
        square=True
    )
    plt.title("Feature Correlation Map with CO2 Consumption", fontsize=14, pad=12)
    plt.tight_layout()

    output_path = csv_path.parent / "co2_correlation_map.png"
    plt.savefig(output_path, dpi=150)
    plt.show()



if __name__ == "__main__":
    generate_correlation_map()

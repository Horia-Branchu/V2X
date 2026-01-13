import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from datacollector.data_collector import baseline_filename, data_dir_name

def main():
    """Generate a Pearson correlation heatmap without time and veh_id and save it"""

    project_root = Path(__file__).resolve().parents[2]
    parquet_path = project_root/ data_dir_name / baseline_filename

    if not parquet_path.exists():
        print(f"Could not find any {baseline_filename} in the project"
              "Run the simulation first so DataCollector generates it")
        return None

    df = pd.read_parquet(parquet_path)
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
    output_path = parquet_path.parent / "co2_correlation_map.png"
    plt.savefig(output_path, dpi=150)
    print("Done")

if __name__ == "__main__":
    print(__file__)
    main()

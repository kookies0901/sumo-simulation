import csv
import os
import pandas as pd
import os

def load_scenario(scenario_id: str, csv_path: str = "data/dataset_1/scenario_matrix.csv") -> dict:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Scenario matrix not found at: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if scenario_id not in df['scenario_id'].values:
        raise ValueError(f"Scenario ID '{scenario_id}' not found in scenario matrix.")

    row = df[df['scenario_id'] == scenario_id].iloc[0].to_dict()

    # Ensure type conversion
    row["vehicle_count"] = int(row["vehicle_count"])
    row["ev_ratio"] = float(row["ev_ratio"])
    row["cs_count"] = int(row["cs_count"])
    row["cs_layout"] = str(row["cs_layout"])
    row["cs_layout_id"] = str(row["cs_layout_id"])
    row["map_id"] = str(row["map_id"])
    row["enable_traci"] = bool(row["enable_traci"])
    row["repeat_n"] = int(row["repeat_n"])

    return row

# For test/debug
if __name__ == "__main__":
    config = load_scenario("v2_gl_10")  # example scenario_id
    for k, v in config.items():
        print(f"{k}: {v}")



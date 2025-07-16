import os
import pandas as pd
from extract_layout_features import extract_layout_features

def build_training_dataset(
    sumo_base_dir: str,
    net_xml_path: str,
    scenario_ids: list,
    output_filename: str = "ml_training_dataset.csv"
):
    """
    构建机器学习训练数据集：结合 layout 特征 + 仿真结果
    """
    rows = []

    for scenario_id in scenario_ids:
        try:
            scenario_path = os.path.join(sumo_base_dir, scenario_id)
            cs_path = os.path.join(scenario_path, "cs")
            output_path = os.path.join(scenario_path, "output")

            # 找到对应 layout 文件
            cs_file = next(
                (f for f in os.listdir(cs_path) if f.endswith(".xml")), None
            )
            if not cs_file:
                print(f"⚠️ No cs XML found for {scenario_id}")
                continue

            cs_xml = os.path.join(cs_path, cs_file)
            layout_features = extract_layout_features(cs_xml, net_xml_path)

            # 读取仿真输出文件
            sim_csv = os.path.join(output_path, "traci_data.csv")
            if not os.path.exists(sim_csv):
                print(f"⚠️ No traci_data.csv found for {scenario_id}")
                continue

            sim_df = pd.read_csv(sim_csv)
            mean_wait = sim_df["charging_wait_time"].mean()
            mean_charge = sim_df["charging_duration"].mean()

            row = {
                "scenario_id": scenario_id,
                "mean_wait_time": mean_wait,
                "mean_charging_time": mean_charge,
                **layout_features
            }

            rows.append(row)

        except Exception as e:
            print(f"❌ Error processing {scenario_id}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(output_filename, index=False)
    print(f"✅ Dataset saved to {output_filename}")
    return df

# 示例调用（需修改路径）
training_df = build_training_dataset(
    sumo_base_dir="/home/ubuntu/project/MSC/Msc_Project/sumo/dataset_1",
    net_xml_path="/home/ubuntu/project/MSC/Msc_Project/data/map/glasgow_clean.net.xml",
    scenario_ids=["S001", "S002", "S003"],
    output_filename="final_ml_dataset.csv"
)

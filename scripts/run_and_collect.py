import os
import pandas as pd
from run_simulation import run_simulation

def main():
    # 1. 读取所有数据集
    with open('data/dataset_list.txt', 'r') as f:
        datasets = [line.strip() for line in f if line.strip()]
    print(f"发现数据集: {datasets}")

    for dataset in datasets:
        print(f"\n==== 处理数据集: {dataset} ====")
        dataset_dir = f"data/{dataset}"
        scenario_csv = f"{dataset_dir}/scenario_matrix.csv"
        # 2. 读取所有scenario_id
        try:
            df = pd.read_csv(scenario_csv)
            scenario_ids = df['scenario_id'].tolist()
        except Exception as e:
            print(f"[读取场景配置] 失败: {e}")
            continue
        # 3. 依次仿真每个场景
        for scenario_id in scenario_ids:
            print(f"\n-- 仿真场景: {scenario_id} --")
            try:
                run_simulation(scenario_id, dataset)
            except Exception as e:
                print(f"[场景 {scenario_id} 仿真失败]: {e}")
                continue
    print("\n==== 所有数据集仿真完成 ====")

if __name__ == "__main__":
    main()

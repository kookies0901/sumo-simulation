import os
import sys
import traceback
import pandas as pd
from clean_net import clean_net
from generate_cs_candidates import generate_cs_candidates
from generator_charging_site import generate_charging_stations
from extract_layout_features import extract_layout_features
from generator_trip import generate_trip_routes
from generate_sumocfg import generate_sumocfg
from load_scenario import load_scenario

# SUMO工具路径、车辆类型定义等可根据实际环境调整
def get_sumo_env():
    sumo_home = os.environ.get('SUMO_HOME', '/usr/share/sumo')
    tools = os.path.join(sumo_home, 'tools')
    duarouter_path = 'duarouter'  # 假设已加入PATH
    vehicles_add = os.path.join('data', 'vehicles.add.xml')
    return tools, duarouter_path, vehicles_add

def main():
    # 1. 读取所有数据集
    with open('data/dataset_list.txt', 'r') as f:
        datasets = [line.strip() for line in f if line.strip()]
    print(f"发现数据集: {datasets}")

    for dataset in datasets:
        print(f"\n==== 处理数据集: {dataset} ====")
        dataset_dir = f"data/{dataset}"
        sumo_dir = f"sumo/{dataset}"
        map_raw = "data/map/glasgow.net.xml"
        map_clean = "data/map/glasgow_clean.net.xml"
        layout_json = f"{dataset_dir}/layout_registry.json"
        scenario_csv = f"{dataset_dir}/scenario_matrix.csv"

        # 2. 地图清洗（如已存在可跳过）
        if not os.path.exists(map_clean):
            try:
                print(f"[地图清洗] {map_raw} -> {map_clean}")
                clean_net(map_raw, map_clean)
            except Exception as e:
                print(f"[地图清洗] 失败: {e}")
                traceback.print_exc()
                continue
        else:
            print(f"[地图清洗] 已存在: {map_clean}")

        # 3. 生成充电桩候选点
        try:
            print(f"[候选点生成] {map_clean} -> {layout_json}")
            # 可根据实际需求调整layout_count/cs_per_layout
            generate_cs_candidates(map_clean, layout_json, layout_count=10, cs_per_layout=215)
        except Exception as e:
            print(f"[候选点生成] 失败: {e}")
            traceback.print_exc()
            continue

        # 4. 读取所有scenario_id
        try:
            df = pd.read_csv(scenario_csv)
            scenario_ids = df['scenario_id'].tolist()
        except Exception as e:
            print(f"[读取场景配置] 失败: {e}")
            traceback.print_exc()
            continue

        # 5. SUMO环境
        tools, duarouter_path, vehicles_add = get_sumo_env()

        # 6. 依次处理每个场景
        for scenario_id in scenario_ids:
            print(f"\n-- 处理场景: {scenario_id} --")
            try:
                # 6.1 生成充电站布局
                generate_charging_stations(
                    scenario_id,
                    layout_path=layout_json,
                    scenario_path=scenario_csv,
                    out_dir_base=sumo_dir
                )
                # 6.2 提取布局特征
                config = load_scenario(scenario_id, scenario_csv)
                cs_layout_id = config['cs_layout_id']
                cs_xml_path = os.path.join(sumo_dir, scenario_id, 'cs', f'{cs_layout_id}.xml')
                net_xml_path = map_clean
                features = extract_layout_features(cs_xml_path, net_xml_path)
                # 写入csv
                feat_dir = os.path.join(sumo_dir, scenario_id, 'cs')
                os.makedirs(feat_dir, exist_ok=True)
                feat_csv = os.path.join(feat_dir, 'layout_features.csv')
                pd.DataFrame([features]).to_csv(feat_csv, index=False)
                print(f"[特征提取] 已写入: {feat_csv}")
                # 6.3 生成出行路线
                routes_dir = os.path.join(sumo_dir, scenario_id, 'routes')
                rou_file = generate_trip_routes(
                    config,
                    sumo_net_file=map_clean,
                    output_dir=routes_dir,
                    sumo_tools_path=tools,
                    vehicles_add_file=vehicles_add,
                    duarouter_path=duarouter_path
                )
                print(f"[路线生成] 已生成: {rou_file}")
                # 6.4 生成sumocfg
                generate_sumocfg(
                    scenario_id,
                    base_dir=sumo_dir,
                    net_file=map_clean
                )
            except Exception as e:
                print(f"[场景 {scenario_id} 处理失败]: {e}")
                traceback.print_exc()
                continue

    print("\n==== 所有数据集处理完成 ====")

if __name__ == "__main__":
    main()

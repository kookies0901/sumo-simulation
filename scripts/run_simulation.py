import sys
import os
# 先加入SUMO tools路径
sys.path.insert(0, "/usr/share/sumo/tools")
import traci
import xml.etree.ElementTree as ET
import pandas as pd
import argparse

def extract_charging_lanes(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lanes = set()
    for cs in root.findall("chargingStation"):
        lane = cs.get("lane")
        if lane:
            lanes.add(lane)
    return lanes

def run_simulation(scenario_id, dataset):
    base_dir = os.path.join("sumo", dataset, scenario_id)
    config_path = os.path.join(base_dir, f"{scenario_id}.sumocfg")
    # 动态查找cs xml
    cs_dir = os.path.join(base_dir, "cs")
    cs_xml = None
    for fname in os.listdir(cs_dir):
        if fname.endswith(".xml"):
            cs_xml = os.path.join(cs_dir, fname)
            break
    if cs_xml is None:
        raise FileNotFoundError(f"No charging station xml found in {cs_dir}")
    charging_lanes = extract_charging_lanes(cs_xml)
    traci.start(["sumo", "-c", config_path])
    charging_records = {}
    def is_ev(veh_id):
        return veh_id.startswith("EV_")
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        for veh_id in traci.vehicle.getIDList():
            if not is_ev(veh_id):
                continue
            if veh_id not in charging_records:
                charging_records[veh_id] = {
                    "waiting": 0,
                    "charging": 0,
                    "was_charging": False
                }
            # 优化：用traci.vehicle.isCharging判断是否正在充电
            try:
                is_charging = traci.vehicle.isCharging(veh_id)
            except Exception:
                is_charging = False
            if is_charging:
                charging_records[veh_id]["was_charging"] = True
                charging_records[veh_id]["charging"] += 1
            elif not charging_records[veh_id]["was_charging"]:
                charging_records[veh_id]["waiting"] += 1
    traci.close()
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "traci_data.csv")
    pd.DataFrame([
        {
            "veh_id": vid,
            "charging_wait_time": v["waiting"],
            "charging_duration": v["charging"]
        }
        for vid, v in charging_records.items()
    ]).to_csv(output_path, index=False)
    print(f"✅ Charging data saved to {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SUMO simulation for a scenario.")
    parser.add_argument("-s", "--scenario_id", required=True, help="Scenario ID, e.g. S001")
    parser.add_argument("-d", "--dataset", required=True, help="Dataset name, e.g. dataset1")
    parser.add_argument("-p", "--parallel", action="store_true", help="[预留] 并行运行（暂未实现）")
    args = parser.parse_args()
    run_simulation(args.scenario_id, args.dataset)

import os
import sys
import traci
import xml.etree.ElementTree as ET
import sumolib

# === 你需要根据实际项目路径调整这几个变量 ===
SUMO_BINARY = "sumo-gui"   # 如果只想后台跑可写 "sumo"
SUMOCFG_PATH = os.path.join("sumo", "base.sumocfg")
CHARGING_XML = os.path.join("sumo", "chargingStations.add.xml")

# 电量低于多少去充电（单位：kWh），你可以自行调整
CHARGING_THRESHOLD = 10.0
CHARGING_DURATION = 600  # 充电桩停留时间，秒

# ========== 自动读取所有充电桩 ==========
def get_charging_stations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    stations = []
    for cs in root.findall("chargingStation"):
        lane = cs.get("lane")
        start = float(cs.get("startPos"))
        end = float(cs.get("endPos"))
        pos = (start + end) / 2
        stations.append((lane, pos))
    return stations

charging_stations = get_charging_stations(CHARGING_XML)

# ========== 辅助函数：找最近桩 ==========
def find_nearest_station(vehicle_lane, vehicle_pos):
    min_dist = float('inf')
    nearest = charging_stations[0]
    for cs_lane, cs_pos in charging_stations:
        try:
            dist = sumolib.miscutils.distance2D(
                traci.lane.getShape(vehicle_lane)[int(vehicle_pos)][0],
                traci.lane.getShape(cs_lane)[int(cs_pos)][0]
            )
        except Exception:
            # Fallback: 只比较lane是否一致，优先本lane，其次返回第一个
            if vehicle_lane == cs_lane:
                return (cs_lane, cs_pos)
            continue
        if dist < min_dist:
            min_dist = dist
            nearest = (cs_lane, cs_pos)
    return nearest

# ========== 主控脚本 ==========
if __name__ == "__main__":
    sumo_cmd = [SUMO_BINARY, "-c", SUMOCFG_PATH, "--start"]

    traci.start(sumo_cmd)
    print("TraCI connected. Monitoring vehicles...")

    ev_goto_station = set()  # 已经被安排去充电的EV（防止重复下达stop指令）

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            for veh_id in traci.vehicle.getIDList():
                if traci.vehicle.getTypeID(veh_id) == "EV":
                    soc = float(traci.vehicle.getParameter(veh_id, "device.battery.actualBatteryCapacity"))
                    if soc < CHARGING_THRESHOLD and veh_id not in ev_goto_station:
                        v_lane = traci.vehicle.getLaneID(veh_id)
                        v_pos = traci.vehicle.getLanePosition(veh_id)
                        cs_lane, cs_pos = find_nearest_station(v_lane, v_pos)
                        print(f"[INFO] EV {veh_id} battery={soc:.2f} kWh, sending to station {cs_lane} @ {cs_pos:.1f}")
                        traci.vehicle.setStop(veh_id, cs_lane, pos=cs_pos, duration=CHARGING_DURATION, flags=0)
                        ev_goto_station.add(veh_id)
    finally:
        traci.close()
        print("TraCI session closed.")

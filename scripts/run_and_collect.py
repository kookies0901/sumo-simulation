import os
import pandas as pd
import xml.etree.ElementTree as ET
import argparse
import sys
import gzip
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_simulation import run_simulation
from vehicle_config import load_vehicle_config

def parse_charging_data(scenario_id, dataset, vehicle_config=None):
    """
    从输出文件中解析充电数据
    """
    # 如果没有提供配置，则加载默认配置
    if vehicle_config is None:
        vehicle_config = load_vehicle_config()
    base_dir = os.path.join("sumo", dataset, scenario_id)
    
    
    # 文件路径
    battery_file = os.path.join(output_dir, "battery_output.xml.gz")
    charging_file = os.path.join(output_dir, "chargingevents.xml.gz")
    summary_file = os.path.join(output_dir, "summary_output.xml.gz")
    
    # 检查文件是否存在
    if not all(os.path.exists(f) for f in [battery_file, charging_file, summary_file]):
        print(f"❌ 输出文件不完整，跳过数据解析: {scenario_id}")
        return None
    
    try:
        # 解析充电事件数据
        charging_data = parse_charging_events(charging_file)
        
        # 解析电池数据
        battery_data = parse_battery_data(battery_file, vehicle_config)
        
        # 解析汇总数据
        summary_data = parse_summary_data(summary_file)
        
        # 合并数据
        result = {
            "scenario_id": scenario_id,
            "dataset": dataset,
            "avg_waiting_time": charging_data.get("avg_waiting_time", 0),
            "avg_charging_time": charging_data.get("avg_charging_time", 0),
            "total_charging_events": charging_data.get("total_events", 0),
            "total_energy_charged": charging_data.get("total_energy", 0),
            "ev_count": battery_data.get("ev_count", 0),
            "avg_initial_soc": battery_data.get("avg_initial_soc", 0),
            "avg_final_soc": battery_data.get("avg_final_soc", 0),
            "simulation_duration": summary_data.get("duration", 0)
        }
        
        print(f"✅ 数据解析完成: {scenario_id}")
        return result
        
    except Exception as e:
        print(f"❌ 数据解析失败 {scenario_id}: {e}")
        return None

def parse_charging_events(charging_file):
    """
    解析充电事件文件
    """
    try:
        # 处理压缩的XML文件
        with gzip.open(charging_file, 'rt', encoding='utf-8') as f:
            tree = ET.parse(f)
            root = tree.getroot()
    except ET.ParseError:
        try:
            # 如果失败，尝试使用更宽松的解析器
            with gzip.open(charging_file, 'rt', encoding='utf-8') as f:
                parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
                tree = ET.parse(f, parser=parser)
                root = tree.getroot()
        except ET.ParseError:
            # 如果还是失败，尝试手动解析
            print(f"⚠️ XML解析失败，尝试手动解析: {charging_file}")
            return parse_charging_events_manual(charging_file)
    
    total_energy = 0
    total_events = 0
    waiting_times = []
    charging_times = []
    
    # 解析充电站数据
    for cs in root.findall("chargingStation"):
        energy = float(cs.get("totalEnergyCharged", 0))
        steps = int(cs.get("chargingSteps", 0))
        
        if energy > 0:
            total_energy += energy
            total_events += steps
            # 假设每个充电步骤代表1秒
            charging_times.append(steps)
    
    # 计算平均值
    avg_charging_time = sum(charging_times) / len(charging_times) if charging_times else 0
    avg_waiting_time = 0  # 需要从其他文件获取等待时间
    
    return {
        "total_energy": total_energy,
        "total_events": total_events,
        "avg_charging_time": avg_charging_time,
        "avg_waiting_time": avg_waiting_time
    }

def parse_charging_events_manual(charging_file):
    """
    手动解析充电事件文件（处理不完整的XML）
    """
    total_energy = 0
    total_events = 0
    
    try:
        with gzip.open(charging_file, 'rt', encoding='utf-8') as f:
            content = f.read()
            
        # 查找所有chargingStation标签
        import re
        pattern = r'<chargingStation id="[^"]*" totalEnergyCharged="([^"]*)" chargingSteps="([^"]*)"'
        matches = re.findall(pattern, content)
        
        for energy_str, steps_str in matches:
            try:
                energy = float(energy_str)
                steps = int(steps_str)
                
                if energy > 0:
                    total_energy += energy
                    total_events += steps
            except ValueError:
                continue
                
    except Exception as e:
        print(f"手动解析失败: {e}")
    
    return {
        "total_energy": total_energy,
        "total_events": total_events,
        "avg_charging_time": total_events if total_events > 0 else 0,
        "avg_waiting_time": 0
    }

def parse_battery_data(battery_file, vehicle_config=None):
    """
    解析电池数据文件
    """
    # 如果没有提供配置，则加载默认配置
    if vehicle_config is None:
        vehicle_config = load_vehicle_config()
    
    max_battery_capacity = vehicle_config["capacity"]
    
    try:
        # 处理压缩的XML文件
        with gzip.open(battery_file, 'rt', encoding='utf-8') as f:
            tree = ET.parse(f)
            root = tree.getroot()
    except ET.ParseError:
        try:
            # 如果失败，尝试使用更宽松的解析器
            with gzip.open(battery_file, 'rt', encoding='utf-8') as f:
                parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
                tree = ET.parse(f, parser=parser)
                root = tree.getroot()
        except ET.ParseError:
            # 如果还是失败，尝试手动解析
            print(f"⚠️ XML解析失败，尝试手动解析: {battery_file}")
            return parse_battery_data_manual(battery_file, vehicle_config)
    
    ev_vehicles = set()
    initial_capacities = {}
    final_capacities = {}
    
    for timestep in root.findall("timestep"):
        for vehicle in timestep.findall("vehicle"):
            veh_id = vehicle.get("id")
            if veh_id.startswith("EV_"):
                ev_vehicles.add(veh_id)
                actual_capacity = float(vehicle.get("chargeLevel", 0))
                max_capacity = float(vehicle.get("capacity", max_battery_capacity))
                
                # 记录初始容量
                if veh_id not in initial_capacities:
                    initial_capacities[veh_id] = actual_capacity
                
                # 更新最终容量
                final_capacities[veh_id] = actual_capacity
    
    # 计算SOC (0-1之间的小数)
    initial_socs = [cap / max_battery_capacity for cap in initial_capacities.values()]
    final_socs = [cap / max_battery_capacity for cap in final_capacities.values()]
    
    return {
        "ev_count": len(ev_vehicles),
        "avg_initial_soc": sum(initial_socs) / len(initial_socs) if initial_socs else 0,
        "avg_final_soc": sum(final_socs) / len(final_socs) if final_socs else 0
    }

def parse_battery_data_manual(battery_file, vehicle_config=None):
    """
    手动解析电池数据文件（处理不完整的XML）
    """
    # 如果没有提供配置，则加载默认配置
    if vehicle_config is None:
        vehicle_config = load_vehicle_config()
    
    max_battery_capacity = vehicle_config["capacity"]
    ev_vehicles = set()
    initial_capacities = {}
    final_capacities = {}
    
    try:
        with gzip.open(battery_file, 'rt', encoding='utf-8') as f:
            content = f.read()
            
        # 查找所有vehicle标签
        import re
        pattern = r'<vehicle id="([^"]*)"[^>]*chargeLevel="([^"]*)"[^>]*capacity="([^"]*)"'
        matches = re.findall(pattern, content)
        
        for veh_id, actual_cap_str, max_cap_str in matches:
            if veh_id.startswith("EV_"):
                try:
                    actual_capacity = float(actual_cap_str)
                    max_capacity = float(max_cap_str)
                    
                    ev_vehicles.add(veh_id)
                    
                    # 记录初始容量
                    if veh_id not in initial_capacities:
                        initial_capacities[veh_id] = actual_capacity
                    
                    # 更新最终容量
                    final_capacities[veh_id] = actual_capacity
                    
                except ValueError:
                    continue
                    
    except Exception as e:
        print(f"手动解析失败: {e}")
    
    # 计算SOC (0-1之间的小数)
    initial_socs = [cap / max_battery_capacity for cap in initial_capacities.values()]
    final_socs = [cap / max_battery_capacity for cap in final_capacities.values()]
    
    return {
        "ev_count": len(ev_vehicles),
        "avg_initial_soc": sum(initial_socs) / len(initial_socs) if initial_socs else 0,
        "avg_final_soc": sum(final_socs) / len(final_socs) if final_socs else 0
    }

def parse_summary_data(summary_file):
    """
    解析汇总数据文件
    """
    try:
        # 处理压缩的XML文件
        with gzip.open(summary_file, 'rt', encoding='utf-8') as f:
            tree = ET.parse(f)
            root = tree.getroot()
    except ET.ParseError:
        try:
            # 如果失败，尝试使用更宽松的解析器
            with gzip.open(summary_file, 'rt', encoding='utf-8') as f:
                parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
                tree = ET.parse(f, parser=parser)
                root = tree.getroot()
        except ET.ParseError:
            # 如果还是失败，尝试手动解析
            print(f"⚠️ XML解析失败，尝试手动解析: {summary_file}")
            return parse_summary_data_manual(summary_file)
    
    # 查找仿真结束时间
    end_time = 0
    for step in root.findall("step"):
        time = float(step.get("time", 0))
        end_time = max(end_time, time)
    
    return {
        "duration": end_time
    }

def parse_summary_data_manual(summary_file):
    """
    手动解析汇总数据文件（处理不完整的XML）
    """
    end_time = 0
    
    try:
        with gzip.open(summary_file, 'rt', encoding='utf-8') as f:
            content = f.read()
            
        # 查找所有step标签
        import re
        pattern = r'<step time="([^"]*)"'
        matches = re.findall(pattern, content)
        
        for time_str in matches:
            try:
                time = float(time_str)
                end_time = max(end_time, time)
            except ValueError:
                continue
                
    except Exception as e:
        print(f"手动解析失败: {e}")
    
    return {
        "duration": end_time
    }

def collect_data_for_scenario(scenario_id, dataset, vehicle_config=None):
    """
    为单个场景运行仿真并收集数据
    """
    print(f"\n🚗 开始处理场景: {scenario_id}")
    
    # 1. 运行仿真
    success = run_simulation(scenario_id, dataset)
    if not success:
        print(f"❌ 仿真失败: {scenario_id}")
        return None
    
    # 2. 收集数据
    data = parse_charging_data(scenario_id, dataset, vehicle_config)
    return data

def collect_data_for_all_scenarios(vehicle_config=None):
    """
    为所有场景运行仿真并收集数据
    """
    # 如果没有提供配置，则加载默认配置
    if vehicle_config is None:
        vehicle_config = load_vehicle_config()
    
    # 1. 读取所有数据集
    with open('data/dataset_list.txt', 'r') as f:
        datasets = [line.strip() for line in f if line.strip()]
    print(f"发现数据集: {datasets}")
    
    all_results = []

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
        
        # 3. 依次处理每个场景
        for scenario_id in scenario_ids:
            data = collect_data_for_scenario(scenario_id, dataset, vehicle_config)
            if data:
                all_results.append(data)
    
    return all_results

def save_results(results, output_file="charging_analysis.csv"):
    """
    保存结果到CSV文件
    """
    if not results:
        print("❌ 没有数据可保存")
        return
    
    # 如果output_file是相对路径，则保存到对应场景的result目录
    if not os.path.isabs(output_file) and len(results) > 0:
        scenario_id = results[0].get('scenario_id')
        dataset = results[0].get('dataset')
        if scenario_id and dataset:
            # 创建result目录
            result_dir = os.path.join("sumo", dataset, scenario_id, "result")
            os.makedirs(result_dir, exist_ok=True)
            
            # 更新输出文件路径
            output_file = os.path.join(result_dir, output_file)
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"✅ 结果已保存到: {output_file}")
    print(f"📊 共处理 {len(results)} 个场景")

def main():
    parser = argparse.ArgumentParser(description="运行SUMO仿真并收集充电数据")
    parser.add_argument("-s", "--scenario_id", help="单个场景ID，例如 S001")
    parser.add_argument("-d", "--dataset", help="数据集名称，例如 dataset_1")
    parser.add_argument("-o", "--output", default="charging_analysis.csv", 
                       help="输出文件名 (默认: charging_analysis.csv)")
    parser.add_argument("--all", action="store_true", help="处理所有场景")
    
    args = parser.parse_args()
    
    # 加载车辆配置
    vehicle_config = load_vehicle_config()
    
    if args.scenario_id and args.dataset:
        # 处理单个场景
        print(f"🎯 处理单个场景: {args.scenario_id}")
        data = collect_data_for_scenario(args.scenario_id, args.dataset, vehicle_config)
        if data:
            save_results([data], args.output)
        else:
            print("❌ 场景处理失败")
    
    elif args.all:
        # 处理所有场景
        print("🌍 处理所有场景")
        results = collect_data_for_all_scenarios(vehicle_config)
        save_results(results, args.output)
    
    else:
        # 默认处理所有场景
        print("🌍 处理所有场景 (默认)")
        results = collect_data_for_all_scenarios(vehicle_config)
        save_results(results, args.output)

if __name__ == "__main__":
    main()

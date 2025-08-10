import os
import sys
import csv
import json
import argparse
import subprocess
import xml.etree.ElementTree as ET
import pandas as pd
import logging
import gzip
from datetime import datetime
from vehicle_config import load_vehicle_config

def load_scenario_matrix(matrix_file):
    """加载场景矩阵"""
    scenarios = []
    with open(matrix_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenarios.append(row)
    return scenarios

def create_sumo_config(scenario_id, rou_file, cs_file, net_file, output_dir):
    """创建SUMO配置文件"""
    # 配置文件应该放在场景根目录，不是output目录
    scenario_dir = os.path.dirname(output_dir)
    config_file = os.path.join(scenario_dir, f"{scenario_id}.sumocfg")
    
    # 创建场景目录
    os.makedirs(scenario_dir, exist_ok=True)
    
    # 配置文件内容
    config_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{os.path.relpath(net_file, scenario_dir)}"/>
        <route-files value="{os.path.relpath(rou_file, scenario_dir)}"/>
        <additional-files value="{os.path.relpath(cs_file, scenario_dir)},{os.path.relpath('data/vehicles.add.xml', scenario_dir)}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="20000"/>
    </time>
    <report>
        <verbose value="true"/>
        <no-step-log value="true"/>
    </report>
    <output>
        <tripinfo-output value="output/tripinfo_output.xml.gz"/>
        <chargingstations-output value="output/chargingevents.xml.gz"/>
        <battery-output value="output/battery_output.xml.gz"/>
        <tripinfo-output.write-unfinished value="true"/>
        <summary-output value="output/summary_output.xml.gz"/>
    </output>
</configuration>"""
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    return config_file

def run_simulation(config_file, output_dir):
    """运行SUMO仿真"""
    try:
        # 使用场景目录作为工作目录
        scenario_dir = os.path.dirname(output_dir)
        config_filename = os.path.basename(config_file)
        
        logging.info(f"🔍 调试信息:")
        logging.info(f"   - 配置文件: {config_file}")
        logging.info(f"   - 场景目录: {scenario_dir}")
        logging.info(f"   - 输出目录: {output_dir}")
        logging.info(f"   - 工作目录: {scenario_dir}")
        logging.info(f"   - 执行命令: /usr/bin/time -v sumo -c {config_filename}")
        
        # 检查配置文件是否存在
        if not os.path.exists(config_file):
            logging.error(f"❌ 配置文件不存在: {config_file}")
            return False
            
        # 检查工作目录是否存在
        if not os.path.exists(scenario_dir):
            logging.error(f"❌ 场景目录不存在: {scenario_dir}")
            return False
        
        # 创建时间日志文件
        time_log_file = os.path.join(output_dir, "sumo.time.log")
        
        # 使用 /usr/bin/time -v 运行仿真并记录详细资源使用情况
        result = subprocess.run(
            ["/usr/bin/time", "-v", "sumo", "-c", config_filename],
            cwd=scenario_dir,
            capture_output=True,
            text=True,
            check=True
        )
        
        # 解析时间输出中的资源使用信息
        time_output = result.stderr  # time 命令的输出在 stderr
        resource_info = {}
        
        for line in time_output.split('\n'):
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                resource_info[key] = value
        
        # 记录资源使用情况
        logging.info(f"✅ 仿真完成")
        logging.info(f"📊 资源使用情况:")
        if 'Maximum resident set size (kbytes)' in resource_info:
            max_memory_kb = resource_info['Maximum resident set size (kbytes)']
            max_memory_mb = float(max_memory_kb) / 1024
            logging.info(f"   - 最大内存使用: {max_memory_mb:.2f} MB ({max_memory_kb} KB)")
        
        if 'User time (seconds)' in resource_info:
            user_time = resource_info['User time (seconds)']
            logging.info(f"   - 用户时间: {user_time} 秒")
        
        if 'System time (seconds)' in resource_info:
            system_time = resource_info['System time (seconds)']
            logging.info(f"   - 系统时间: {system_time} 秒")
        
        if 'Elapsed (wall clock) time (h:mm:ss or m:ss)' in resource_info:
            elapsed_time = resource_info['Elapsed (wall clock) time (h:mm:ss or m:ss)']
            logging.info(f"   - 总运行时间: {elapsed_time}")
        
        # 保存详细的时间日志
        with open(time_log_file, 'w', encoding='utf-8') as f:
            f.write("=== SUMO 仿真资源使用详情 ===\n")
            f.write(f"配置文件: {config_file}\n")
            f.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("=== 标准输出 ===\n")
            f.write(result.stdout)
            f.write("\n\n=== 资源使用详情 ===\n")
            f.write(time_output)
        
        logging.info(f"📄 详细日志保存到: {time_log_file}")
        logging.info(f"📄 标准输出: {result.stdout[:500]}...")  # 只显示前500字符
        
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ 仿真失败:")
        logging.error(f"   返回码: {e.returncode}")
        logging.error(f"   标准输出: {e.stdout}")
        logging.error(f"   错误输出: {e.stderr}")
        
        # 即使失败也尝试解析资源使用信息
        if e.stderr:
            time_output = e.stderr
            resource_info = {}
            for line in time_output.split('\n'):
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    resource_info[key] = value
            
            if 'Maximum resident set size (kbytes)' in resource_info:
                max_memory_kb = resource_info['Maximum resident set size (kbytes)']
                max_memory_mb = float(max_memory_kb) / 1024
                logging.error(f"   最大内存使用: {max_memory_mb:.2f} MB ({max_memory_kb} KB)")
        
        return False
    except Exception as e:
        logging.error(f"❌ 其他错误: {e}")
        return False

def parse_charging_data(output_dir, vehicle_config):
    """解析充电数据"""
    battery_file = os.path.join(output_dir, "battery_output.xml.gz")
    charging_events_file = os.path.join(output_dir, "chargingevents.xml.gz")
    summary_file = os.path.join(output_dir, "summary_output.xml.gz")
    
    data = {
        'avg_waiting_time': 0.0,
        'avg_charging_time': 0.0,
        'ev_count': 0,
        'avg_initial_soc': 0.0,
        'avg_final_soc': 0.0,
        'simulation_duration': 0.0
    }
    
    # 解析电池数据
    if os.path.exists(battery_file):
        try:
            # 处理压缩的XML文件
            with gzip.open(battery_file, 'rt', encoding='utf-8') as f:
                tree = ET.parse(f)
                root = tree.getroot()
            
            ev_vehicles = set()
            initial_capacities = {}
            final_capacities = {}
            max_battery_capacity = vehicle_config["capacity"]
            
            for timestep in root.findall("timestep"):
                for vehicle in timestep.findall("vehicle"):
                    veh_id = vehicle.get("id")
                    if veh_id.startswith("EV_"):
                        ev_vehicles.add(veh_id)
                        actual_capacity = float(vehicle.get("chargeLevel", 0))
                        
                        if veh_id not in initial_capacities:
                            initial_capacities[veh_id] = actual_capacity
                        final_capacities[veh_id] = actual_capacity
            
            if initial_capacities and final_capacities:
                initial_socs = [cap / max_battery_capacity for cap in initial_capacities.values()]
                final_socs = [cap / max_battery_capacity for cap in final_capacities.values()]
                
                data['ev_count'] = len(ev_vehicles)
                data['avg_initial_soc'] = sum(initial_socs) / len(initial_socs) if initial_socs else 0.0
                data['avg_final_soc'] = sum(final_socs) / len(final_socs) if final_socs else 0.0
                
        except Exception as e:
            print(f"⚠️ 解析电池数据失败: {e}")
    
    # 解析充电事件数据
    if os.path.exists(charging_events_file):
        try:
            # 处理压缩的XML文件
            with gzip.open(charging_events_file, 'rt', encoding='utf-8') as f:
                tree = ET.parse(f)
                root = tree.getroot()
            
            waiting_times = []
            charging_times = []
            
            for event in root.findall("chargingEvent"):
                waiting_time = float(event.get("waitingTime", 0))
                charging_time = float(event.get("chargingTime", 0))
                
                if waiting_time > 0:
                    waiting_times.append(waiting_time)
                if charging_time > 0:
                    charging_times.append(charging_time)
            
            if waiting_times:
                data['avg_waiting_time'] = sum(waiting_times) / len(waiting_times)
            if charging_times:
                data['avg_charging_time'] = sum(charging_times) / len(charging_times)
                
        except Exception as e:
            print(f"⚠️ 解析充电事件数据失败: {e}")
    
    # 解析摘要数据
    if os.path.exists(summary_file):
        try:
            # 处理压缩的XML文件
            with gzip.open(summary_file, 'rt', encoding='utf-8') as f:
                tree = ET.parse(f)
                root = tree.getroot()
            
            for step in root.findall("step"):
                data['simulation_duration'] = float(step.get("time", 0))
                break
                
        except Exception as e:
            print(f"⚠️ 解析摘要数据失败: {e}")
    
    return data

def save_results(results, output_file):
    """保存结果到CSV文件"""
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"✅ 结果保存到: {output_file}")

def run_single_scenario(scenario_id, cs_layout_id, rou_type, data_dir, output_dir):
    """运行单个场景"""
    logging.info(f"🎯 运行场景: {scenario_id} ({cs_layout_id} + {rou_type})")
    
    # 文件路径
    rou_file = os.path.join(data_dir, "routes", f"{rou_type}.rou.xml")
    cs_file = os.path.join(data_dir, "cs", f"{cs_layout_id}.xml")
    net_file = os.path.join(data_dir, "map", "glasgow_clean.net.xml")
    
    logging.info(f"📁 文件路径:")
    logging.info(f"   - 路由文件: {rou_file}")
    logging.info(f"   - 充电站文件: {cs_file}")
    logging.info(f"   - 网络文件: {net_file}")
    
    # 检查文件是否存在
    if not os.path.exists(rou_file):
        logging.error(f"❌ 路由文件不存在: {rou_file}")
        return None
    if not os.path.exists(cs_file):
        logging.error(f"❌ 充电站文件不存在: {cs_file}")
        return None
    if not os.path.exists(net_file):
        logging.error(f"❌ 网络文件不存在: {net_file}")
        return None
    
    # 创建场景输出目录
    scenario_output_dir = os.path.join(output_dir, scenario_id, "output")
    os.makedirs(scenario_output_dir, exist_ok=True)
    logging.info(f"   - 输出目录: {scenario_output_dir}")
    
    # 创建SUMO配置
    config_file = create_sumo_config(scenario_id, rou_file, cs_file, net_file, scenario_output_dir)
    
    # 运行仿真
    success = run_simulation(config_file, scenario_output_dir)
    
    if success:
        # 解析数据
        vehicle_config = load_vehicle_config()
        data = parse_charging_data(scenario_output_dir, vehicle_config)
        
        # 添加场景信息
        data['scenario_id'] = scenario_id
        data['cs_layout_id'] = cs_layout_id
        data['rou_type'] = rou_type
        
        return data
    else:
        return None

def run_all_scenarios(matrix_file, data_dir, output_dir):
    """运行所有场景"""
    print(f"🚀 开始运行所有场景")
    
    # 加载场景矩阵
    scenarios = load_scenario_matrix(matrix_file)
    print(f"📊 加载了 {len(scenarios)} 个场景")
    
    # 运行所有场景
    results = []
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] 处理场景: {scenario['scenario_id']}")
        
        data = run_single_scenario(
            scenario['scenario_id'],
            scenario['cs_layout_id'],
            scenario['rou_type'],
            data_dir,
            output_dir
        )
        
        if data:
            results.append(data)
            print(f"✅ 场景 {scenario['scenario_id']} 完成")
        else:
            print(f"❌ 场景 {scenario['scenario_id']} 失败")
    
    # 保存结果
    if results:
        result_file = os.path.join(output_dir, "charging_analysis.csv")
        save_results(results, result_file)
        print(f"\n🎉 所有场景运行完成！共成功运行 {len(results)} 个场景")
    else:
        print(f"\n❌ 没有成功运行的场景")

def main():
    # 设置日志
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"simulation_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"🚀 开始运行仿真，日志文件: {log_file}")
    
    parser = argparse.ArgumentParser(description='运行全局仿真')
    parser.add_argument('--matrix', type=str, 
                       default='data/scenario_matrix.csv',
                       help='场景矩阵文件路径')
    parser.add_argument('--data_dir', type=str, 
                       default='data',
                       help='数据目录路径')
    parser.add_argument('--output_dir', type=str, 
                       default='sumo',
                       help='输出目录路径')
    parser.add_argument('-s', '--scenario', type=str,
                       help='运行单个场景 (格式: S001)')
    
    args = parser.parse_args()
    
    if args.scenario:
        # 运行单个场景
        scenarios = load_scenario_matrix(args.matrix)
        target_scenario = None
        
        for scenario in scenarios:
            if scenario['scenario_id'] == args.scenario:
                target_scenario = scenario
                break
        
        if target_scenario:
            data = run_single_scenario(
                target_scenario['scenario_id'],
                target_scenario['cs_layout_id'],
                target_scenario['rou_type'],
                args.data_dir,
                args.output_dir
            )
            
            if data:
                # 保存单个场景结果
                result_file = os.path.join(args.output_dir, target_scenario['scenario_id'], "result", "charging_analysis.csv")
                os.makedirs(os.path.dirname(result_file), exist_ok=True)
                save_results([data], result_file)
                print(f"✅ 单个场景运行完成: {args.scenario}")
            else:
                print(f"❌ 单个场景运行失败: {args.scenario}")
        else:
            print(f"❌ 未找到场景: {args.scenario}")
    else:
        # 运行所有场景
        run_all_scenarios(args.matrix, args.data_dir, args.output_dir)

if __name__ == '__main__':
    main() 
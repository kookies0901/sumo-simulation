#!/usr/bin/env python3
"""
分析SUMO输出文件 - 重构版本
只解压 tripinfo_output.xml.gz 和 chargingevents.xml.gz，分析8个关键指标
支持批量处理多个场景
"""

import os
import sys
import csv
import json
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import logging
import gzip
import shutil
import numpy as np
from datetime import datetime
from vehicle_config import load_vehicle_config

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_scenario_matrix(matrix_file):
    """加载场景矩阵文件"""
    try:
        scenarios = []
        with open(matrix_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                scenarios.append(row)
        logging.info(f"📊 从场景矩阵加载了 {len(scenarios)} 个场景")
        return scenarios
    except Exception as e:
        logging.error(f"❌ 加载场景矩阵失败: {e}")
        return []

def get_scenario_list(start_id="S001", end_id="S050"):
    """生成场景ID列表，支持S001-S050格式"""
    try:
        # 提取数字部分
        start_num = int(start_id[1:])
        end_num = int(end_id[1:])
        
        scenarios = []
        for i in range(start_num, end_num + 1):
            scenario_id = f"S{i:03d}"
            scenarios.append(scenario_id)
        
        logging.info(f"📊 生成了 {len(scenarios)} 个场景ID: {start_id} 到 {end_id}")
        return scenarios
    except Exception as e:
        logging.error(f"❌ 生成场景列表失败: {e}")
        return []

def decompress_file(gz_file_path, xml_file_path):
    """解压 .xml.gz 文件到 .xml 文件"""
    try:
        with gzip.open(gz_file_path, 'rb') as f_in:
            with open(xml_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        logging.info(f"✅ 解压完成: {gz_file_path} -> {xml_file_path}")
        return True
    except Exception as e:
        logging.error(f"❌ 解压失败 {gz_file_path}: {e}")
        return False

def calculate_statistics(values, name="数据"):
    """计算统计指标：mean, median, p90"""
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0}
    
    values = np.array(values)
    mean = np.mean(values)
    median = np.median(values)
    p90 = np.percentile(values, 90)
    
    logging.info(f"📊 {name}统计: mean={mean:.2f}, median={median:.2f}, p90={p90:.2f}")
    return {"mean": mean, "median": median, "p90": p90}

def calculate_gini_coefficient(values):
    """计算基尼系数"""
    if not values or len(values) == 0:
        return 0.0
    
    values = np.array(values)
    if np.sum(values) == 0:
        return 0.0
    
    # 排序并计算累积份额
    sorted_values = np.sort(values)
    n = len(sorted_values)
    cumsum = np.cumsum(sorted_values)
    
    # 计算基尼系数
    gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    return gini

def calculate_hhi(values):
    """计算HHI指数 (Herfindahl-Hirschman Index)"""
    if not values or len(values) == 0:
        return 0.0
    
    values = np.array(values)
    total = np.sum(values)
    if total == 0:
        return 0.0
    
    # 计算市场份额的平方和
    market_shares = values / total
    hhi = np.sum(market_shares ** 2)
    return hhi

def calculate_cv(values):
    """计算变异系数 (Coefficient of Variation)"""
    if not values or len(values) == 0:
        return 0.0
    
    values = np.array(values)
    mean = np.mean(values)
    if mean == 0:
        return 0.0
    
    std = np.std(values)
    cv = std / mean
    return cv

def parse_tripinfo_data(xml_file_path):
    """解析tripinfo数据，获取车辆duration、waitingTime、rerouteNo等信息"""
    logging.info(f"🔍 开始解析tripinfo数据: {xml_file_path}")
    
    data = {
        'durations': [],
        'waiting_times': [],
        'reroute_count': 0,
        'ev_charging_failures': 0,
        'total_vehicles': 0
    }
    
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        for tripinfo in root.findall("tripinfo"):
            data['total_vehicles'] += 1
            
            # 获取duration
            duration = float(tripinfo.get("duration", 0))
            data['durations'].append(duration)
            
            # 获取waitingTime
            waiting_time = float(tripinfo.get("waitingTime", 0))
            data['waiting_times'].append(waiting_time)
            
            # 统计reroute次数
            reroute_no = int(tripinfo.get("rerouteNo", 0))
            if reroute_no > 0:
                data['reroute_count'] += 1
        
        # 统计stationfinder元素数量（充电失败的EV）
        # stationfinder是tripinfo的子元素，需要递归查找
        stationfinder_count = 0
        for tripinfo in root.findall("tripinfo"):
            stationfinders = tripinfo.findall("stationfinder")
            stationfinder_count += len(stationfinders)
        data['ev_charging_failures'] = stationfinder_count
        
        logging.info(f"📊 解析完成: {data['total_vehicles']} 辆车")
        logging.info(f"📊 重新路由车辆数: {data['reroute_count']}")
        logging.info(f"📊 EV充电失败数: {data['ev_charging_failures']}")
        
    except Exception as e:
        logging.error(f"❌ 解析tripinfo数据失败: {e}")
    
    return data

def parse_charging_events_data(xml_file_path):
    """解析chargingevents数据，获取充电桩使用情况"""
    logging.info(f"🔍 开始解析chargingevents数据: {xml_file_path}")
    
    data = {
        'charging_steps': [],
        'total_energy_charged': [],
        'charging_vehicles_count': [],
        'ev_charging_participation': set(),
        'total_charging_stations': 0,
        'used_charging_stations': 0
    }
    
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        for station in root.findall("chargingStation"):
            data['total_charging_stations'] += 1
            
            # 获取充电时间
            charging_steps = int(station.get("chargingSteps", 0))
            if charging_steps > 0:
                data['charging_steps'].append(charging_steps)
            
            # 获取总充电量
            total_energy = float(station.get("totalEnergyCharged", 0))
            data['total_energy_charged'].append(total_energy)
            
            if total_energy > 0:
                data['used_charging_stations'] += 1
            
            # 统计充电车辆数
            vehicle_count = len(station.findall("vehicle"))
            data['charging_vehicles_count'].append(vehicle_count)
            
            # 统计参与充电的EV
            for vehicle in station.findall("vehicle"):
                veh_id = vehicle.get("id", "")
                if veh_id.startswith("EV_"):
                    data['ev_charging_participation'].add(veh_id)
        
        logging.info(f"📊 解析完成: {data['total_charging_stations']} 个充电桩")
        logging.info(f"📊 使用过的充电桩: {data['used_charging_stations']}")
        logging.info(f"📊 参与充电的EV数: {len(data['ev_charging_participation'])}")
        
    except Exception as e:
        logging.error(f"❌ 解析chargingevents数据失败: {e}")
    
    return data

def analyze_scenario_metrics(scenario_id, output_dir):
    """分析场景的8个关键指标"""
    logging.info(f"🎯 开始分析场景: {scenario_id}")
    
    # 定义文件路径
    gz_files = {
        'tripinfo': os.path.join(output_dir, "tripinfo_output.xml.gz"),
        'charging': os.path.join(output_dir, "chargingevents.xml.gz")
    }
    
    xml_files = {
        'tripinfo': os.path.join(output_dir, "tripinfo_output_temp.xml"),
        'charging': os.path.join(output_dir, "chargingevents_temp.xml")
    }
    
    # 检查压缩文件是否存在
    missing_files = []
    for name, gz_path in gz_files.items():
        if not os.path.exists(gz_path):
            missing_files.append(gz_path)
    
    if missing_files:
        logging.error(f"❌ 缺少压缩文件:")
        for file in missing_files:
            logging.error(f"   - {file}")
        return None
    
    # 解压文件
    logging.info("🔧 开始解压文件...")
    for name, gz_path in gz_files.items():
        xml_path = xml_files[name]
        if not decompress_file(gz_path, xml_path):
            logging.error(f"❌ 解压失败，无法继续分析")
            return None
    
    # 解析数据
    tripinfo_data = parse_tripinfo_data(xml_files['tripinfo'])
    charging_data = parse_charging_events_data(xml_files['charging'])
    
    # 计算8个关键指标
    metrics = {}
    
    # 1. 平均duration（车辆口径）
    duration_stats = calculate_statistics(tripinfo_data['durations'], "车辆行驶时间")
    metrics.update({
        'duration_mean': duration_stats['mean'],
        'duration_median': duration_stats['median'],
        'duration_p90': duration_stats['p90']
    })
    
    # 2. 平均充电时间（事件口径）
    charging_time_stats = calculate_statistics(charging_data['charging_steps'], "充电时间")
    metrics.update({
        'charging_time_mean': charging_time_stats['mean'],
        'charging_time_median': charging_time_stats['median'],
        'charging_time_p90': charging_time_stats['p90']
    })
    
    # 3. 平均等待时间
    waiting_time_stats = calculate_statistics(tripinfo_data['waiting_times'], "等待时间")
    metrics.update({
        'waiting_time_mean': waiting_time_stats['mean'],
        'waiting_time_median': waiting_time_stats['median'],
        'waiting_time_p90': waiting_time_stats['p90']
    })
    
    # 4. 站点"充电量"的离散程度
    energy_values = [e for e in charging_data['total_energy_charged'] if e > 0]
    if energy_values:
        energy_gini = calculate_gini_coefficient(energy_values)
        energy_cv = calculate_cv(energy_values)
        energy_hhi = calculate_hhi(energy_values)
        energy_p90_p50_ratio = np.percentile(energy_values, 90) / np.percentile(energy_values, 50) if len(energy_values) > 0 else 0
        zero_usage_rate = (charging_data['total_charging_stations'] - len(energy_values)) / charging_data['total_charging_stations']
    else:
        energy_gini = energy_cv = energy_hhi = energy_p90_p50_ratio = 0.0
        zero_usage_rate = 1.0
    
    metrics.update({
        'energy_gini': energy_gini,
        'energy_cv': energy_cv,
        'energy_hhi': energy_hhi,
        'energy_p90_p50_ratio': energy_p90_p50_ratio,
        'energy_zero_usage_rate': zero_usage_rate
    })
    
    # 5. "充电桩的充电车辆数"的离散程度
    vehicle_count_values = [v for v in charging_data['charging_vehicles_count'] if v > 0]
    if vehicle_count_values:
        vehicle_gini = calculate_gini_coefficient(vehicle_count_values)
        vehicle_cv = calculate_cv(vehicle_count_values)
        vehicle_hhi = calculate_hhi(vehicle_count_values)
        vehicle_zero_usage_rate = (charging_data['total_charging_stations'] - len(vehicle_count_values)) / charging_data['total_charging_stations']
    else:
        vehicle_gini = vehicle_cv = vehicle_hhi = 0.0
        vehicle_zero_usage_rate = 1.0
    
    metrics.update({
        'vehicle_gini': vehicle_gini,
        'vehicle_cv': vehicle_cv,
        'vehicle_hhi': vehicle_hhi,
        'vehicle_zero_usage_rate': vehicle_zero_usage_rate
    })
    
    # 6. 充电桩使用覆盖率
    coverage_rate = charging_data['used_charging_stations'] / charging_data['total_charging_stations']
    metrics['charging_station_coverage'] = coverage_rate
    
    # 7. reroute数
    metrics['reroute_count'] = tripinfo_data['reroute_count']
    
    # 8. EV充电参与率
    ev_total = 1800  # 固定值
    ev_participation_rate = len(charging_data['ev_charging_participation']) / ev_total
    metrics['ev_charging_participation_rate'] = ev_participation_rate
    
    # 9. 充电失败的EV数
    metrics['ev_charging_failures'] = tripinfo_data['ev_charging_failures']
    
    # 添加场景信息
    metrics['scenario_id'] = scenario_id
    
    # 清理临时文件
    logging.info("🧹 清理临时文件...")
    for xml_path in xml_files.values():
        if os.path.exists(xml_path):
            os.remove(xml_path)
    
    logging.info("✅ 分析完成")
    return metrics

def save_results(metrics, output_file):
    """保存结果到CSV文件"""
    df = pd.DataFrame([metrics])
    df.to_csv(output_file, index=False)
    logging.info(f"✅ 结果保存到: {output_file}")

def check_scenario_files(scenario_id, output_dir):
    """检查场景所需的文件是否存在"""
    scenario_output_dir = os.path.join(output_dir, scenario_id, "output")
    
    required_files = [
        os.path.join(scenario_output_dir, "tripinfo_output.xml.gz"),
        os.path.join(scenario_output_dir, "chargingevents.xml.gz")
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    return len(missing_files) == 0, missing_files

def batch_analyze_scenarios(scenario_ids, output_dir, result_dir=None):
    """批量分析多个场景"""
    logging.info(f"🚀 开始批量分析 {len(scenario_ids)} 个场景")
    
    # 首先检查所有场景的文件可用性
    logging.info("🔍 检查场景文件可用性...")
    available_scenarios = []
    missing_scenarios = []
    
    for scenario_id in scenario_ids:
        files_exist, missing_files = check_scenario_files(scenario_id, output_dir)
        if files_exist:
            available_scenarios.append(scenario_id)
        else:
            missing_scenarios.append((scenario_id, missing_files))
            logging.warning(f"⚠️ 场景 {scenario_id} 缺少文件:")
            for missing_file in missing_files:
                logging.warning(f"   - {missing_file}")
    
    logging.info(f"📊 文件检查结果: {len(available_scenarios)}/{len(scenario_ids)} 个场景可用")
    
    if not available_scenarios:
        logging.error("❌ 没有可用的场景进行分析")
        return []
    
    all_results = []
    success_count = 0
    failure_count = 0
    
    for i, scenario_id in enumerate(available_scenarios, 1):
        logging.info(f"\n[{i}/{len(available_scenarios)}] 处理场景: {scenario_id}")
        
        try:
            # 构建单个场景的输出路径
            scenario_output_dir = os.path.join(output_dir, scenario_id, "output")
            
            # 分析单个场景
            metrics = analyze_scenario_metrics(scenario_id, scenario_output_dir)
            
            if metrics:
                all_results.append(metrics)
                success_count += 1
                logging.info(f"✅ 场景 {scenario_id} 分析完成")
            else:
                failure_count += 1
                logging.error(f"❌ 场景 {scenario_id} 分析失败")
                
        except Exception as e:
            failure_count += 1
            logging.error(f"❌ 场景 {scenario_id} 处理异常: {e}")
            continue
    
    # 保存批量结果
    if all_results:
        if result_dir is None:
            result_dir = os.path.join(output_dir, "batch_results")
        
        os.makedirs(result_dir, exist_ok=True)
        
        # 保存所有结果到CSV
        batch_result_file = os.path.join(result_dir, "batch_charging_analysis.csv")
        df = pd.DataFrame(all_results)
        df.to_csv(batch_result_file, index=False)
        
        # 保存汇总统计
        summary_file = os.path.join(result_dir, "batch_summary.csv")
        summary_data = {
            'total_scenarios_requested': len(scenario_ids),
            'scenarios_with_files': len(available_scenarios),
            'scenarios_missing_files': len(missing_scenarios),
            'successful_analyses': success_count,
            'failed_analyses': failure_count,
            'success_rate_of_available': success_count / len(available_scenarios) if available_scenarios else 0,
            'overall_success_rate': success_count / len(scenario_ids),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        summary_df = pd.DataFrame([summary_data])
        summary_df.to_csv(summary_file, index=False)
        
        logging.info(f"\n🎉 批量分析完成！")
        logging.info(f"📊 请求处理场景数: {len(scenario_ids)}")
        logging.info(f"📁 有文件的场景数: {len(available_scenarios)}")
        logging.info(f"📂 缺少文件的场景数: {len(missing_scenarios)}")
        logging.info(f"✅ 分析成功: {success_count}")
        logging.info(f"❌ 分析失败: {failure_count}")
        logging.info(f"📈 可用场景成功率: {success_count/len(available_scenarios)*100:.1f}%" if available_scenarios else "📈 可用场景成功率: 0%")
        logging.info(f"📊 总体成功率: {success_count/len(scenario_ids)*100:.1f}%")
        logging.info(f"💾 结果保存到: {batch_result_file}")
        logging.info(f"📋 汇总保存到: {summary_file}")
        
        return all_results
    else:
        logging.error("❌ 没有成功分析的场景")
        return []

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description='分析SUMO输出文件的8个关键指标，支持批量处理')
    parser.add_argument('--scenario_id', type=str,
                       help='单个场景ID (例如: S001)')
    parser.add_argument('--output_dir', type=str, 
                       default='sumo',
                       help='输出目录路径')
    parser.add_argument('--result_dir', type=str,
                       help='结果保存目录 (默认: output_dir/scenario_id/result)')
    
    # 批量处理参数
    parser.add_argument('--batch', action='store_true',
                       help='启用批量处理模式')
    parser.add_argument('--start_id', type=str, default='S001',
                       help='批量处理起始场景ID (默认: S001)')
    parser.add_argument('--end_id', type=str, default='S050',
                       help='批量处理结束场景ID (默认: S050)')
    parser.add_argument('--matrix', type=str,
                       help='场景矩阵文件路径，用于批量处理')
    parser.add_argument('--all', action='store_true',
                       help='处理所有可用场景 (自动检测)')
    
    args = parser.parse_args()
    
    # 检查参数组合
    if args.batch or args.start_id != 'S001' or args.end_id != 'S050' or args.matrix or args.all:
        # 批量处理模式
        if args.matrix:
            # 从场景矩阵文件读取场景列表
            scenarios = load_scenario_matrix(args.matrix)
            if scenarios:
                scenario_ids = [s['scenario_id'] for s in scenarios]
                logging.info(f"📊 从场景矩阵读取到 {len(scenario_ids)} 个场景")
            else:
                logging.error("❌ 无法从场景矩阵读取场景列表")
                sys.exit(1)
        elif args.all:
            # 自动检测所有可用场景
            scenario_ids = []
            for item in os.listdir(args.output_dir):
                if os.path.isdir(os.path.join(args.output_dir, item)) and item.startswith('S') and len(item) == 4:
                    try:
                        int(item[1:])  # 验证格式
                        scenario_ids.append(item)
                    except ValueError:
                        continue
            
            scenario_ids.sort()
            logging.info(f"📊 自动检测到 {len(scenario_ids)} 个场景")
        else:
            # 使用指定的场景范围
            scenario_ids = get_scenario_list(args.start_id, args.end_id)
        
        if not scenario_ids:
            logging.error("❌ 没有找到要处理的场景")
            sys.exit(1)
        
        # 执行批量分析
        batch_analyze_scenarios(scenario_ids, args.output_dir, args.result_dir)
        
    elif args.scenario_id:
        # 单个场景处理模式
        # 构建路径
        scenario_output_dir = os.path.join(args.output_dir, args.scenario_id, "output")
        
        if args.result_dir:
            result_dir = args.result_dir
        else:
            result_dir = os.path.join(args.output_dir, args.scenario_id, "result")
        
        # 创建结果目录
        os.makedirs(result_dir, exist_ok=True)
        
        # 分析数据
        metrics = analyze_scenario_metrics(args.scenario_id, scenario_output_dir)
        
        if metrics:
            # 保存结果
            result_file = os.path.join(result_dir, "charging_analysis.csv")
            save_results(metrics, result_file)
            
            # 打印结果摘要
            print("\n" + "="*60)
            print("📊 8个关键指标分析结果摘要")
            print("="*60)
            print(f"场景ID: {metrics['scenario_id']}")
            print(f"\n1. 车辆行驶时间 (秒):")
            print(f"   - 平均: {metrics['duration_mean']:.2f}")
            print(f"   - 中位数: {metrics['duration_median']:.2f}")
            print(f"   - P90: {metrics['duration_p90']:.2f}")
            
            print(f"\n2. 充电时间 (秒):")
            print(f"   - 平均: {metrics['charging_time_mean']:.2f}")
            print(f"   - 中位数: {metrics['charging_time_median']:.2f}")
            print(f"   - P90: {metrics['charging_time_p90']:.2f}")
            
            print(f"\n3. 等待时间 (秒):")
            print(f"   - 平均: {metrics['waiting_time_mean']:.2f}")
            print(f"   - 中位数: {metrics['waiting_time_median']:.2f}")
            print(f"   - P90: {metrics['waiting_time_p90']:.2f}")
            
            print(f"\n4. 充电量离散程度:")
            print(f"   - 基尼系数: {metrics['energy_gini']:.4f}")
            print(f"   - 变异系数: {metrics['energy_cv']:.4f}")
            print(f"   - HHI指数: {metrics['energy_hhi']:.4f}")
            print(f"   - P90/P50比: {metrics['energy_p90_p50_ratio']:.4f}")
            print(f"   - 零使用率: {metrics['energy_zero_usage_rate']:.4f}")
            
            print(f"\n5. 充电车辆数离散程度:")
            print(f"   - 基尼系数: {metrics['vehicle_gini']:.4f}")
            print(f"   - 变异系数: {metrics['vehicle_cv']:.4f}")
            print(f"   - HHI指数: {metrics['vehicle_hhi']:.4f}")
            print(f"   - 零使用率: {metrics['vehicle_zero_usage_rate']:.4f}")
            
            print(f"\n6. 充电桩使用覆盖率: {metrics['charging_station_coverage']:.4f}")
            print(f"7. 重新路由车辆数: {metrics['reroute_count']}")
            print(f"8. EV充电参与率: {metrics['ev_charging_participation_rate']:.4f}")
            print(f"9. EV充电失败数: {metrics['ev_charging_failures']}")
            print("="*60)
        else:
            logging.error("❌ 分析失败")
            sys.exit(1)
    else:
        # 默认显示帮助信息
        parser.print_help()
        print("\n💡 使用示例:")
        print("   # 分析单个场景")
        print("   python analyze_compressed_output.py --scenario_id S001")
        print("\n   # 批量处理S001-S050")
        print("   python analyze_compressed_output.py --batch --start_id S001 --end_id S050")
        print("\n   # 从场景矩阵文件批量处理")
        print("   python analyze_compressed_output.py --matrix data/scenario_matrix.csv")
        print("\n   # 自动检测并处理所有场景")
        print("   python analyze_compressed_output.py --all")

if __name__ == '__main__':
    main() 
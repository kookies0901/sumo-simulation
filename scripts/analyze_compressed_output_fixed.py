#!/usr/bin/env python3
"""
分析SUMO输出文件 - 修复浮点精度问题版本
解决duration_p90和waiting_time_p90的尾数精度问题
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
from decimal import Decimal, ROUND_HALF_UP

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def safe_float_convert(value_str, default=0.0, precision=6):
    """安全的浮点数转换，避免精度问题"""
    try:
        # 先转为字符串，再用Decimal进行精确计算
        if isinstance(value_str, str):
            # 移除可能的科学计数法和多余精度
            decimal_val = Decimal(value_str)
        else:
            decimal_val = Decimal(str(value_str))
        
        # 四舍五入到指定精度
        rounded_decimal = decimal_val.quantize(
            Decimal('0.' + '0' * precision), 
            rounding=ROUND_HALF_UP
        )
        
        return float(rounded_decimal)
    except:
        return default

def calculate_statistics_fixed(values, name="数据", precision=4):
    """计算统计指标：mean, median, p90 - 修复精度问题版本"""
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0}
    
    # 转换为numpy数组并清理精度
    clean_values = []
    for v in values:
        if isinstance(v, (int, float)) and not np.isnan(v) and not np.isinf(v):
            # 使用round而不是Decimal以保持兼容性，但限制精度
            clean_val = round(float(v), precision)
            clean_values.append(clean_val)
    
    if not clean_values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0}
    
    values_array = np.array(clean_values)
    
    # 计算统计量并四舍五入
    mean = round(float(np.mean(values_array)), precision)
    median = round(float(np.median(values_array)), precision)
    p90 = round(float(np.percentile(values_array, 90)), precision)
    
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
    return round(gini, 6)

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
    return round(hhi, 6)

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
    return round(cv, 6)

def parse_tripinfo_data_fixed(xml_file_path):
    """解析tripinfo数据，修复精度问题"""
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
            
            # 获取duration - 使用安全转换
            duration_str = tripinfo.get("duration", "0")
            duration = safe_float_convert(duration_str, 0.0)
            data['durations'].append(duration)
            
            # 获取waitingTime - 使用安全转换
            waiting_time_str = tripinfo.get("waitingTime", "0")
            waiting_time = safe_float_convert(waiting_time_str, 0.0)
            data['waiting_times'].append(waiting_time)
            
            # 统计reroute次数
            reroute_no = int(tripinfo.get("rerouteNo", 0))
            if reroute_no > 0:
                data['reroute_count'] += 1
        
        # 统计stationfinder元素数量（充电失败的EV）
        stationfinder_count = 0
        for tripinfo in root.findall("tripinfo"):
            stationfinders = tripinfo.findall("stationfinder")
            stationfinder_count += len(stationfinders)
        data['ev_charging_failures'] = stationfinder_count
        
        logging.info(f"📊 解析完成: {data['total_vehicles']} 辆车")
        logging.info(f"📊 重新路由车辆数: {data['reroute_count']}")
        logging.info(f"📊 EV充电失败数: {data['ev_charging_failures']}")
        
        # 数据质量检查
        duration_info = f"Duration范围: {min(data['durations']):.2f} - {max(data['durations']):.2f}"
        waiting_info = f"WaitingTime范围: {min(data['waiting_times']):.2f} - {max(data['waiting_times']):.2f}"
        logging.info(f"📊 {duration_info}")
        logging.info(f"📊 {waiting_info}")
        
    except Exception as e:
        logging.error(f"❌ 解析tripinfo数据失败: {e}")
    
    return data

def parse_charging_events_data_fixed(xml_file_path):
    """解析chargingevents数据，修复精度问题"""
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
            
            # 获取充电时间 - 使用安全转换
            charging_steps_str = station.get("chargingSteps", "0")
            charging_steps = safe_float_convert(charging_steps_str, 0.0)
            if charging_steps > 0:
                data['charging_steps'].append(charging_steps)
            
            # 获取总充电量 - 使用安全转换
            total_energy_str = station.get("totalEnergyCharged", "0")
            total_energy = safe_float_convert(total_energy_str, 0.0)
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

def analyze_scenario_metrics_fixed(scenario_id, output_dir):
    """分析场景的8个关键指标 - 修复精度问题版本"""
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
    tripinfo_data = parse_tripinfo_data_fixed(xml_files['tripinfo'])
    charging_data = parse_charging_events_data_fixed(xml_files['charging'])
    
    # 计算8个关键指标 - 使用修复的函数
    metrics = {}
    
    # 1. 平均duration（车辆口径）
    duration_stats = calculate_statistics_fixed(tripinfo_data['durations'], "车辆行驶时间")
    metrics.update({
        'duration_mean': duration_stats['mean'],
        'duration_median': duration_stats['median'],
        'duration_p90': duration_stats['p90']
    })
    
    # 2. 平均充电时间（事件口径）
    charging_time_stats = calculate_statistics_fixed(charging_data['charging_steps'], "充电时间")
    metrics.update({
        'charging_time_mean': charging_time_stats['mean'],
        'charging_time_median': charging_time_stats['median'],
        'charging_time_p90': charging_time_stats['p90']
    })
    
    # 3. 平均等待时间
    waiting_time_stats = calculate_statistics_fixed(tripinfo_data['waiting_times'], "等待时间")
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
        energy_p90_p50_ratio = round(np.percentile(energy_values, 90) / np.percentile(energy_values, 50), 4) if len(energy_values) > 0 else 0
        zero_usage_rate = round((charging_data['total_charging_stations'] - len(energy_values)) / charging_data['total_charging_stations'], 6)
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
        vehicle_zero_usage_rate = round((charging_data['total_charging_stations'] - len(vehicle_count_values)) / charging_data['total_charging_stations'], 6)
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
    coverage_rate = round(charging_data['used_charging_stations'] / charging_data['total_charging_stations'], 6)
    metrics['charging_station_coverage'] = coverage_rate
    
    # 7. reroute数
    metrics['reroute_count'] = tripinfo_data['reroute_count']
    
    # 8. EV充电参与率
    ev_total = 1800  # 固定值
    ev_participation_rate = round(len(charging_data['ev_charging_participation']) / ev_total, 6)
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
    
    # 输出精度检查信息
    logging.info("🔍 精度检查:")
    logging.info(f"   Duration P90: {metrics['duration_p90']}")
    logging.info(f"   Waiting Time P90: {metrics['waiting_time_p90']}")
    logging.info(f"   Charging Time P90: {metrics['charging_time_p90']}")
    
    logging.info("✅ 分析完成")
    return metrics

def save_results(metrics, output_file):
    """保存结果到CSV文件"""
    df = pd.DataFrame([metrics])
    df.to_csv(output_file, index=False)
    logging.info(f"✅ 结果保存到: {output_file}")

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description='分析SUMO输出文件的8个关键指标 - 修复精度问题版本')
    parser.add_argument('--scenario_id', type=str, required=True,
                       help='场景ID (例如: S001)')
    parser.add_argument('--output_dir', type=str, 
                       default='sumo',
                       help='输出目录路径')
    parser.add_argument('--result_dir', type=str,
                       help='结果保存目录 (默认: output_dir/scenario_id/result)')
    
    args = parser.parse_args()
    
    # 构建路径
    scenario_output_dir = os.path.join(args.output_dir, args.scenario_id, "output")
    
    if args.result_dir:
        result_dir = args.result_dir
    else:
        result_dir = os.path.join(args.output_dir, args.scenario_id, "result")
    
    # 创建结果目录
    os.makedirs(result_dir, exist_ok=True)
    
    # 分析数据
    metrics = analyze_scenario_metrics_fixed(args.scenario_id, scenario_output_dir)
    
    if metrics:
        # 保存结果
        result_file = os.path.join(result_dir, "charging_analysis_fixed.csv")
        save_results(metrics, result_file)
        
        # 打印结果摘要
        print("\n" + "="*60)
        print("📊 8个关键指标分析结果摘要 (修复精度版本)")
        print("="*60)
        print(f"场景ID: {metrics['scenario_id']}")
        print(f"\n1. 车辆行驶时间 (秒):")
        print(f"   - 平均: {metrics['duration_mean']:.4f}")
        print(f"   - 中位数: {metrics['duration_median']:.4f}")
        print(f"   - P90: {metrics['duration_p90']:.4f}")
        
        print(f"\n3. 等待时间 (秒):")
        print(f"   - 平均: {metrics['waiting_time_mean']:.4f}")
        print(f"   - 中位数: {metrics['waiting_time_median']:.4f}")
        print(f"   - P90: {metrics['waiting_time_p90']:.4f}")
        
        print("="*60)
        print("✅ 精度问题已修复，不再出现长尾数")
    else:
        logging.error("❌ 分析失败")
        sys.exit(1)

if __name__ == '__main__':
    main()



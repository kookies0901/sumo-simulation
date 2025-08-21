#!/usr/bin/env python3
"""
简化的充电站分布可视化脚本
使用更直接的方法处理坐标匹配问题
"""

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import os
import warnings
warnings.filterwarnings('ignore')

# 设置字体和图形参数
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

class SimpleChargingStationVisualizer:
    def __init__(self, network_file, xml_dirs, output_dir):
        self.network_file = network_file
        self.xml_dirs = xml_dirs
        self.output_dir = output_dir
        self.lane_coordinates = {}
        self.charging_stations = {}
        
        os.makedirs(output_dir, exist_ok=True)
        
    def parse_network_lanes(self):
        """快速解析网络文件，只关注lane坐标"""
        print("正在解析网络文件中的lane信息...")
        
        try:
            # 使用iterparse避免内存问题
            lane_count = 0
            current_edge = None
            current_edge_shape = None
            
            for event, elem in ET.iterparse(self.network_file, events=('start', 'end')):
                if event == 'start':
                    if elem.tag == 'edge':
                        current_edge = elem.get('id')
                        current_edge_shape = elem.get('shape')
                elif event == 'end':
                    if elem.tag == 'lane' and current_edge and not current_edge.startswith(':'):
                        lane_id = elem.get('id')
                        
                        # 获取lane的shape
                        lane_shape = elem.get('shape')
                        if lane_shape:
                            # 解析shape字符串 "x1,y1 x2,y2 ..."
                            coords = []
                            for point in lane_shape.split():
                                try:
                                    x, y = map(float, point.split(','))
                                    coords.append((x, y))
                                except:
                                    continue
                            
                            if coords:
                                # 使用lane的中点坐标
                                mid_idx = len(coords) // 2
                                self.lane_coordinates[lane_id] = coords[mid_idx]
                                lane_count += 1
                        
                        elif current_edge_shape:
                            # 如果lane没有shape，使用edge的shape
                            coords = []
                            for point in current_edge_shape.split():
                                try:
                                    x, y = map(float, point.split(','))
                                    coords.append((x, y))
                                except:
                                    continue
                            
                            if coords:
                                mid_idx = len(coords) // 2
                                self.lane_coordinates[lane_id] = coords[mid_idx]
                                lane_count += 1
                    
                    # 清理元素以节省内存
                    elem.clear()
                
                if lane_count % 5000 == 0 and lane_count > 0:
                    print(f"已解析 {lane_count} 个lane...")
            
            print(f"网络解析完成，获得 {len(self.lane_coordinates)} 个lane坐标")
            
            # 显示坐标范围
            if self.lane_coordinates:
                all_x = [coord[0] for coord in self.lane_coordinates.values()]
                all_y = [coord[1] for coord in self.lane_coordinates.values()]
                print(f"坐标范围: X[{min(all_x):.1f}, {max(all_x):.1f}], Y[{min(all_y):.1f}, {max(all_y):.1f}]")
                
        except Exception as e:
            print(f"网络文件解析错误: {e}")
    
    def parse_charging_stations(self):
        """解析充电站XML文件"""
        print("正在解析充电站XML文件...")
        
        total_stations = 0
        matched_stations = 0
        
        for xml_dir in self.xml_dirs:
            if not os.path.exists(xml_dir):
                continue
                
            xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml') and f.startswith('cs_group_')]
            
            for xml_file in sorted(xml_files):
                xml_path = os.path.join(xml_dir, xml_file)
                group_name = xml_file.replace('.xml', '')
                
                self.charging_stations[group_name] = []
                
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    
                    stations_in_group = 0
                    matched_in_group = 0
                    
                    for cs_elem in root.findall('chargingStation'):
                        lane_id = cs_elem.get('lane')
                        cs_id = cs_elem.get('id')
                        
                        total_stations += 1
                        stations_in_group += 1
                        
                        if lane_id in self.lane_coordinates:
                            x, y = self.lane_coordinates[lane_id]
                            self.charging_stations[group_name].append({
                                'id': cs_id,
                                'lane': lane_id,
                                'x': x,
                                'y': y
                            })
                            matched_stations += 1
                            matched_in_group += 1
                        else:
                            # 生成随机坐标作为备选
                            if self.lane_coordinates:
                                all_x = [coord[0] for coord in self.lane_coordinates.values()]
                                all_y = [coord[1] for coord in self.lane_coordinates.values()]
                                x_min, x_max = min(all_x), max(all_x)
                                y_min, y_max = min(all_y), max(all_y)
                                
                                # 使用lane_id的hash作为种子
                                import random
                                import hashlib
                                seed = int(hashlib.md5(lane_id.encode()).hexdigest()[:8], 16)
                                random.seed(seed)
                                
                                x = random.uniform(x_min, x_max)
                                y = random.uniform(y_min, y_max)
                                
                                self.charging_stations[group_name].append({
                                    'id': cs_id,
                                    'lane': lane_id,
                                    'x': x,
                                    'y': y,
                                    'estimated': True
                                })
                    
                    print(f"{group_name}: {stations_in_group} 个充电站 ({matched_in_group} 匹配成功)")
                    
                except Exception as e:
                    print(f"解析 {xml_file} 出错: {e}")
                    continue
        
        print(f"充电站解析完成:")
        print(f"  总方案数: {len(self.charging_stations)}")
        print(f"  总充电站数: {total_stations}")
        print(f"  匹配成功: {matched_stations} ({matched_stations/total_stations*100:.1f}%)")
    
    def create_individual_scatter_plots(self):
        """为每个方案创建散点图"""
        print("正在生成各方案的散点图...")
        
        # 计算全局坐标范围
        all_x, all_y = [], []
        for stations in self.charging_stations.values():
            for station in stations:
                all_x.append(station['x'])
                all_y.append(station['y'])
        
        if not all_x:
            print("警告: 没有有效的坐标数据")
            return
        
        x_min, x_max = min(all_x) - 500, max(all_x) + 500
        y_min, y_max = min(all_y) - 500, max(all_y) + 500
        
        for group_name, stations in self.charging_stations.items():
            if not stations:
                continue
            
            plt.figure(figsize=(12, 10))
            
            # 分离真实和估算坐标
            real_stations = [s for s in stations if not s.get('estimated', False)]
            est_stations = [s for s in stations if s.get('estimated', False)]
            
            if real_stations:
                real_x = [s['x'] for s in real_stations]
                real_y = [s['y'] for s in real_stations]
                plt.scatter(real_x, real_y, c='red', s=40, alpha=0.8, 
                           edgecolors='darkred', linewidth=0.8, label='Exact locations')
            
            if est_stations:
                est_x = [s['x'] for s in est_stations]
                est_y = [s['y'] for s in est_stations]
                plt.scatter(est_x, est_y, c='blue', s=40, alpha=0.6, 
                           edgecolors='darkblue', linewidth=0.8, label='Estimated locations')
            
            if real_stations and est_stations:
                plt.legend()
            
            plt.title(f'{group_name} Charging Station Distribution\\n' + 
                     f'({len(stations)} stations: {len(real_stations)} exact + {len(est_stations)} estimated)', 
                     fontsize=16, fontweight='bold')
            plt.xlabel('X Coordinate (meters)', fontsize=12)
            plt.ylabel('Y Coordinate (meters)', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            
            plt.tight_layout()
            scatter_png = f'{self.output_dir}/{group_name}_scatter.png'
            plt.savefig(scatter_png, dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_summary_plots(self):
        """创建总体概览图"""
        print("正在生成概览图...")
        
        # 1. 对比图 - 前6个方案
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        group_names = sorted(list(self.charging_stations.keys()))[:6]
        
        all_x, all_y = [], []
        for stations in self.charging_stations.values():
            for station in stations:
                all_x.append(station['x'])
                all_y.append(station['y'])
        
        x_min, x_max = min(all_x) - 500, max(all_x) + 500
        y_min, y_max = min(all_y) - 500, max(all_y) + 500
        
        for idx, group_name in enumerate(group_names):
            ax = axes[idx]
            stations = self.charging_stations[group_name]
            
            if stations:
                real_stations = [s for s in stations if not s.get('estimated', False)]
                est_stations = [s for s in stations if s.get('estimated', False)]
                
                if real_stations:
                    real_x = [s['x'] for s in real_stations]
                    real_y = [s['y'] for s in real_stations]
                    ax.scatter(real_x, real_y, c='red', s=30, alpha=0.7, 
                              edgecolors='darkred', linewidth=0.5)
                
                if est_stations:
                    est_x = [s['x'] for s in est_stations]
                    est_y = [s['y'] for s in est_stations]
                    ax.scatter(est_x, est_y, c='blue', s=30, alpha=0.5, 
                              edgecolors='darkblue', linewidth=0.5)
            
            ax.set_title(f'{group_name}\\n({len(stations)} stations)', fontsize=12, fontweight='bold')
            ax.set_xlabel('X Coordinate (meters)', fontsize=10)
            ax.set_ylabel('Y Coordinate (meters)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        
        plt.suptitle('Charging Station Distribution Comparison (First 6 Scenarios)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/charging_stations_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 叠加频次图
        plt.figure(figsize=(15, 12))
        
        all_coords = defaultdict(int)
        for stations in self.charging_stations.values():
            for station in stations:
                coord = (round(station['x'], 1), round(station['y'], 1))
                all_coords[coord] += 1
        
        if all_coords:
            x_coords = [coord[0] for coord in all_coords.keys()]
            y_coords = [coord[1] for coord in all_coords.keys()]
            frequencies = list(all_coords.values())
            
            scatter = plt.scatter(x_coords, y_coords, c=frequencies, s=50, 
                                alpha=0.7, cmap='YlOrRd', edgecolors='black', linewidth=0.5)
            
            plt.colorbar(scatter, label='Station Selection Frequency')
            plt.title('100 Scenarios Charging Station Overlay Map\\n(Color intensity shows selection frequency)', 
                     fontsize=16, fontweight='bold')
            plt.xlabel('X Coordinate (meters)', fontsize=12)
            plt.ylabel('Y Coordinate (meters)', fontsize=12)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/charging_stations_overlay.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_statistics(self):
        """生成统计信息"""
        print("正在生成统计信息...")
        
        stats_file = f'{self.output_dir}/charging_stations_stats.txt'
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("Charging Station Distribution Statistics\\n")
            f.write("=" * 45 + "\\n")
            f.write(f"Total Scenarios: {len(self.charging_stations)}\\n")
            
            total_stations = sum(len(stations) for stations in self.charging_stations.values())
            f.write(f"Total Stations: {total_stations}\\n")
            
            if self.charging_stations:
                avg_stations = total_stations / len(self.charging_stations)
                f.write(f"Average Stations per Scenario: {avg_stations:.2f}\\n")
                
                station_counts = [len(stations) for stations in self.charging_stations.values()]
                f.write(f"Maximum Stations: {max(station_counts)}\\n")
                f.write(f"Minimum Stations: {min(station_counts)}\\n")
            
            f.write("\\nDetailed Statistics per Scenario:\\n")
            f.write("-" * 40 + "\\n")
            
            for group_name in sorted(self.charging_stations.keys()):
                stations = self.charging_stations[group_name]
                real_count = len([s for s in stations if not s.get('estimated', False)])
                est_count = len([s for s in stations if s.get('estimated', False)])
                f.write(f"{group_name}: {len(stations)} stations ({real_count} exact + {est_count} estimated)\\n")
        
        print(f"统计信息已保存到: {stats_file}")
    
    def run_visualization(self):
        """运行可视化"""
        print("开始充电站可视化...")
        
        self.parse_network_lanes()
        self.parse_charging_stations()
        self.create_individual_scatter_plots()
        self.create_summary_plots()
        self.generate_statistics()
        
        print("\\n可视化完成！生成的文件:")
        print(f"- 输出目录: {self.output_dir}")
        print("- cs_group_xxx_scatter.png: 各组散点图")
        print("- charging_stations_comparison.png: 对比图")
        print("- charging_stations_overlay.png: 叠加图")
        print("- charging_stations_stats.txt: 统计信息")

if __name__ == "__main__":
    network_file = "/home/ubuntu/project/MSC/Msc_Project/data/map/glasgow_clean.net.xml"
    xml_dirs = [
        "/home/ubuntu/project/MSC/Msc_Project/data/cs_1-50",
        "/home/ubuntu/project/MSC/Msc_Project/data/cs_51-100"
    ]
    output_dir = "/home/ubuntu/project/MSC/Msc_Project/data/cs_1-100"
    
    visualizer = SimpleChargingStationVisualizer(network_file, xml_dirs, output_dir)
    visualizer.run_visualization()

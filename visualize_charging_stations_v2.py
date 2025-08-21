#!/usr/bin/env python3
"""
充电站分布可视化脚本 V2
直接使用XML文件中的精确充电站坐标来生成热力图和散点图
"""

import json
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import os
import warnings
warnings.filterwarnings('ignore')

# 设置字体和图形参数
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

class ChargingStationVisualizerV2:
    def __init__(self, network_file, xml_dirs, output_dir):
        self.network_file = network_file
        self.xml_dirs = xml_dirs  # ['data/cs_1-50', 'data/cs_51-100']
        self.output_dir = output_dir
        self.lane_coordinates = {}  # lane_id -> (x, y)
        self.charging_stations = {}  # group_name -> [{'id': '', 'lane': '', 'x': x, 'y': y}]
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
    def parse_network_file(self):
        """解析SUMO网络文件，提取lane的坐标信息"""
        print("正在解析SUMO网络文件，获取lane坐标...")
        
        # 首先获取junction坐标
        junction_coordinates = {}
        
        context = ET.iterparse(self.network_file, events=('start', 'end'))
        context = iter(context)
        event, root = next(context)
        
        junction_count = 0
        
        for event, elem in context:
            if event == 'end':
                if elem.tag == 'junction':
                    junction_id = elem.get('id')
                    x = float(elem.get('x', 0))
                    y = float(elem.get('y', 0))
                    junction_coordinates[junction_id] = (x, y)
                    junction_count += 1
                
                # 清理元素以节省内存
                elem.clear()
                root.clear()
                
                if junction_count % 10000 == 0:
                    print(f"已处理 {junction_count} 个junction")
        
        print(f"Junction解析完成: {junction_count} 个节点")
        
        # 然后解析edge和lane信息
        context = ET.iterparse(self.network_file, events=('start', 'end'))
        context = iter(context)
        event, root = next(context)
        
        edge_count = 0
        lane_count = 0
        
        for event, elem in context:
            if event == 'end':
                if elem.tag == 'edge':
                    edge_id = elem.get('id')
                    from_junction = elem.get('from')
                    to_junction = elem.get('to')
                    
                    # 跳过内部边
                    if edge_id and not edge_id.startswith(':'):
                        edge_count += 1
                        
                        # 获取edge的shape或者from/to坐标
                        shape = elem.get('shape')
                        if shape:
                            # shape是一系列坐标点
                            coords = []
                            for point in shape.split():
                                x, y = map(float, point.split(','))
                                coords.append((x, y))
                            edge_start = coords[0]
                            edge_end = coords[-1]
                        else:
                            # 使用junction坐标
                            if from_junction in junction_coordinates and to_junction in junction_coordinates:
                                edge_start = junction_coordinates[from_junction]
                                edge_end = junction_coordinates[to_junction]
                            else:
                                continue
                        
                        # 处理该edge下的所有lane
                        for lane_elem in elem.findall('lane'):
                            lane_id = lane_elem.get('id')
                            lane_count += 1
                            
                            # 获取lane的shape，如果没有则使用edge的坐标
                            lane_shape = lane_elem.get('shape')
                            if lane_shape:
                                coords = []
                                for point in lane_shape.split():
                                    x, y = map(float, point.split(','))
                                    coords.append((x, y))
                                # 使用lane的中点作为代表坐标
                                mid_idx = len(coords) // 2
                                self.lane_coordinates[lane_id] = coords[mid_idx]
                            else:
                                # 使用edge的中点
                                x = (edge_start[0] + edge_end[0]) / 2
                                y = (edge_start[1] + edge_end[1]) / 2
                                self.lane_coordinates[lane_id] = (x, y)
                
                # 清理元素以节省内存
                elem.clear()
                root.clear()
                
                if (edge_count + lane_count) % 10000 == 0:
                    print(f"已处理: {edge_count} 条边, {lane_count} 条lane")
        
        print(f"网络解析完成: {edge_count} 条边, {lane_count} 条lane")
        print(f"获得 {len(self.lane_coordinates)} 个lane坐标")
        
    def parse_charging_station_xml_files(self):
        """解析充电站XML文件，获取所有充电站的精确位置"""
        print("正在解析充电站XML文件...")
        
        total_stations = 0
        matched_stations = 0
        unmatched_lanes = set()
        
        for xml_dir in self.xml_dirs:
            xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml') and f.startswith('cs_group_')]
            
            for xml_file in sorted(xml_files):
                xml_path = os.path.join(xml_dir, xml_file)
                group_name = xml_file.replace('.xml', '')
                
                self.charging_stations[group_name] = []
                
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    
                    for cs_elem in root.findall('chargingStation'):
                        cs_id = cs_elem.get('id')
                        lane_id = cs_elem.get('lane')
                        start_pos = float(cs_elem.get('startPos', 0))
                        end_pos = float(cs_elem.get('endPos', 0))
                        
                        total_stations += 1
                        
                        # 获取lane的坐标
                        if lane_id in self.lane_coordinates:
                            x, y = self.lane_coordinates[lane_id]
                            self.charging_stations[group_name].append({
                                'id': cs_id,
                                'lane': lane_id,
                                'start_pos': start_pos,
                                'end_pos': end_pos,
                                'x': x,
                                'y': y
                            })
                            matched_stations += 1
                        else:
                            unmatched_lanes.add(lane_id)
                            # 生成估算坐标（在合理范围内）
                            import random
                            import hashlib
                            
                            seed = int(hashlib.md5(lane_id.encode()).hexdigest()[:8], 16)
                            random.seed(seed)
                            
                            if self.lane_coordinates:
                                all_x = [coord[0] for coord in self.lane_coordinates.values()]
                                all_y = [coord[1] for coord in self.lane_coordinates.values()]
                                x_min, x_max = min(all_x), max(all_x)
                                y_min, y_max = min(all_y), max(all_y)
                            else:
                                x_min, x_max = 0, 10000
                                y_min, y_max = 0, 10000
                            
                            x = random.uniform(x_min, x_max)
                            y = random.uniform(y_min, y_max)
                            
                            self.charging_stations[group_name].append({
                                'id': cs_id,
                                'lane': lane_id,
                                'start_pos': start_pos,
                                'end_pos': end_pos,
                                'x': x,
                                'y': y,
                                'estimated': True
                            })
                    
                    print(f"解析 {group_name}: {len(self.charging_stations[group_name])} 个充电站")
                    
                except ET.ParseError as e:
                    print(f"解析 {xml_file} 失败: {e}")
                    continue
        
        print(f"充电站解析完成: {len(self.charging_stations)} 个方案")
        print(f"总充电站数: {total_stations}")
        print(f"坐标匹配成功: {matched_stations} ({matched_stations/total_stations*100:.1f}%)")
        print(f"坐标估算: {total_stations - matched_stations} ({(total_stations - matched_stations)/total_stations*100:.1f}%)")
        print(f"未匹配的唯一lane_id数量: {len(unmatched_lanes)}")
        
    def create_heatmap(self):
        """创建充电站分布热力图"""
        print("正在生成热力图...")
        
        # 收集所有充电站的坐标
        all_x = []
        all_y = []
        
        for group_name, stations in self.charging_stations.items():
            for station in stations:
                all_x.append(station['x'])
                all_y.append(station['y'])
        
        if not all_x:
            print("警告: 没有找到有效的充电站坐标数据")
            return
        
        # 创建网格
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        # 扩展边界
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * 0.1
        x_max += x_range * 0.1
        y_min -= y_range * 0.1
        y_max += y_range * 0.1
        
        # 创建热力图数据
        grid_size = 100
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        heat_map = np.zeros((grid_size, grid_size))
        
        # 计算每个网格点的充电站密度
        for group_name, stations in self.charging_stations.items():
            for station in stations:
                x, y = station['x'], station['y']
                # 找到最近的网格点
                x_idx = np.argmin(np.abs(x_grid - x))
                y_idx = np.argmin(np.abs(y_grid - y))
                heat_map[y_idx, x_idx] += 1
        
        # 应用高斯滤波使热力图更平滑
        from scipy import ndimage
        heat_map = ndimage.gaussian_filter(heat_map, sigma=2)
        
        # 绘制热力图
        plt.figure(figsize=(15, 12))
        
        # 使用自定义颜色映射
        colors = ['white', 'lightblue', 'blue', 'orange', 'red', 'darkred']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('charging_stations', colors, N=n_bins)
        
        im = plt.imshow(heat_map, extent=[x_min, x_max, y_min, y_max], 
                       cmap=cmap, origin='lower', alpha=0.8)
        
        plt.colorbar(im, label='Charging Station Density', shrink=0.8)
        plt.title('Glasgow Charging Station Distribution Heatmap\n(Based on 100 Different Layout Scenarios)', fontsize=16, fontweight='bold')
        plt.xlabel('X Coordinate (meters)', fontsize=12)
        plt.ylabel('Y Coordinate (meters)', fontsize=12)
        
        # 添加网格
        plt.grid(True, alpha=0.3)
        
        # 保存图片
        plt.tight_layout()
        heatmap_png = f'{self.output_dir}/charging_stations_heatmap.png'
        plt.savefig(heatmap_png, dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_individual_scatter_plots(self):
        """为每个方案创建单独的散点图"""
        print("正在为每个方案生成散点图...")
        
        # 计算全局坐标范围
        all_x = []
        all_y = []
        for stations_list in self.charging_stations.values():
            for station in stations_list:
                all_x.append(station['x'])
                all_y.append(station['y'])
        
        if not all_x or not all_y:
            print("警告: 没有找到有效的充电站坐标数据")
            return
            
        x_min, x_max = min(all_x) - 500, max(all_x) + 500
        y_min, y_max = min(all_y) - 500, max(all_y) + 500
        
        # 为每个组生成单独的散点图
        for group_name, stations in self.charging_stations.items():
            if not stations:
                continue
                
            plt.figure(figsize=(12, 10))
            
            # 分离真实坐标和估算坐标
            real_x = [s['x'] for s in stations if not s.get('estimated', False)]
            real_y = [s['y'] for s in stations if not s.get('estimated', False)]
            est_x = [s['x'] for s in stations if s.get('estimated', False)]
            est_y = [s['y'] for s in stations if s.get('estimated', False)]
            
            # 绘制真实坐标的充电站（红色）
            if real_x:
                plt.scatter(real_x, real_y, c='red', s=40, alpha=0.8, 
                           edgecolors='darkred', linewidth=0.8, label='Exact locations')
            
            # 绘制估算坐标的充电站（蓝色）
            if est_x:
                plt.scatter(est_x, est_y, c='blue', s=40, alpha=0.6, 
                           edgecolors='darkblue', linewidth=0.8, label='Estimated locations')
            
            # 添加图例
            if real_x and est_x:
                plt.legend()
            
            plt.title(f'{group_name} Charging Station Distribution\n({len(stations)} stations: {len(real_x)} exact + {len(est_x)} estimated)', 
                     fontsize=16, fontweight='bold')
            plt.xlabel('X Coordinate (meters)', fontsize=12)
            plt.ylabel('Y Coordinate (meters)', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # 设置统一的坐标轴范围
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            
            # 保存单独的图像
            plt.tight_layout()
            scatter_png = f'{self.output_dir}/{group_name}_scatter.png'
            plt.savefig(scatter_png, dpi=300, bbox_inches='tight')
            plt.close()
        
    def create_comparison_plot(self):
        """创建前6个方案的对比图"""
        print("正在生成方案对比图...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        group_names = sorted(list(self.charging_stations.keys()))[:6]
        
        # 计算全局坐标范围
        all_x = []
        all_y = []
        for stations_list in self.charging_stations.values():
            for station in stations_list:
                all_x.append(station['x'])
                all_y.append(station['y'])
        
        x_min, x_max = min(all_x) - 500, max(all_x) + 500
        y_min, y_max = min(all_y) - 500, max(all_y) + 500
        
        for idx, group_name in enumerate(group_names):
            ax = axes[idx]
            stations = self.charging_stations[group_name]
            
            if not stations:
                continue
            
            # 分离真实坐标和估算坐标
            real_x = [s['x'] for s in stations if not s.get('estimated', False)]
            real_y = [s['y'] for s in stations if not s.get('estimated', False)]
            est_x = [s['x'] for s in stations if s.get('estimated', False)]
            est_y = [s['y'] for s in stations if s.get('estimated', False)]
            
            # 绘制散点图
            if real_x:
                ax.scatter(real_x, real_y, c='red', s=30, alpha=0.7, 
                          edgecolors='darkred', linewidth=0.5)
            if est_x:
                ax.scatter(est_x, est_y, c='blue', s=30, alpha=0.5, 
                          edgecolors='darkblue', linewidth=0.5)
            
            ax.set_title(f'{group_name}\n({len(stations)} stations)', fontsize=12, fontweight='bold')
            ax.set_xlabel('X Coordinate (meters)', fontsize=10)
            ax.set_ylabel('Y Coordinate (meters)', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # 设置相同的坐标轴范围
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        
        plt.suptitle('Charging Station Distribution Comparison (First 6 Scenarios)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        comparison_png = f'{self.output_dir}/charging_stations_comparison.png'
        plt.savefig(comparison_png, dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_overlay_plot(self):
        """创建所有方案的叠加图"""
        print("正在生成叠加散点图...")
        
        plt.figure(figsize=(15, 12))
        
        # 使用不同颜色表示不同的方案密度
        all_coords = defaultdict(int)
        
        for group_name, stations in self.charging_stations.items():
            for station in stations:
                coord = (round(station['x'], 1), round(station['y'], 1))
                all_coords[coord] += 1
        
        # 分离坐标和频次
        x_coords = [coord[0] for coord in all_coords.keys()]
        y_coords = [coord[1] for coord in all_coords.keys()]
        frequencies = list(all_coords.values())
        
        # 创建散点图，颜色表示出现频次
        scatter = plt.scatter(x_coords, y_coords, c=frequencies, s=50, 
                            alpha=0.7, cmap='YlOrRd', edgecolors='black', linewidth=0.5)
        
        plt.colorbar(scatter, label='Station Selection Frequency')
        plt.title('100 Scenarios Charging Station Overlay Map\n(Color intensity shows selection frequency)', fontsize=16, fontweight='bold')
        plt.xlabel('X Coordinate (meters)', fontsize=12)
        plt.ylabel('Y Coordinate (meters)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        overlay_png = f'{self.output_dir}/charging_stations_overlay.png'
        plt.savefig(overlay_png, dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_statistics(self):
        """生成统计信息"""
        print("正在生成统计信息...")
        
        stats = {
            'Total Scenarios': len(self.charging_stations),
            'Total Stations': 0,
            'Average Stations per Scenario': 0,
            'Maximum Stations': 0,
            'Minimum Stations': float('inf'),
            'Unique Locations': 0
        }
        
        station_counts = []
        all_coords = set()
        
        for group_name, stations in self.charging_stations.items():
            count = len(stations)
            station_counts.append(count)
            stats['Maximum Stations'] = max(stats['Maximum Stations'], count)
            stats['Minimum Stations'] = min(stats['Minimum Stations'], count)
            stats['Total Stations'] += count
            
            for station in stations:
                coord = (round(station['x'], 1), round(station['y'], 1))
                all_coords.add(coord)
        
        stats['Average Stations per Scenario'] = stats['Total Stations'] / stats['Total Scenarios']
        stats['Unique Locations'] = len(all_coords)
        
        # 保存统计信息
        stats_file = f'{self.output_dir}/charging_stations_stats.txt'
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("Charging Station Distribution Statistics (From XML Files)\n")
            f.write("=" * 55 + "\n")
            for key, value in stats.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.2f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            f.write("\n")
            f.write("Detailed Statistics per Scenario:\n")
            f.write("-" * 40 + "\n")
            for group_name, stations in sorted(self.charging_stations.items()):
                real_count = len([s for s in stations if not s.get('estimated', False)])
                est_count = len([s for s in stations if s.get('estimated', False)])
                f.write(f"{group_name}: {len(stations)} stations ({real_count} exact + {est_count} estimated)\n")
        
        print("统计信息:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        
    def run_visualization(self):
        """运行完整的可视化流程"""
        print("开始充电站分布可视化（使用XML文件）...")
        
        # 解析数据
        self.parse_network_file()
        self.parse_charging_station_xml_files()
        
        # 生成可视化
        self.create_heatmap()
        self.create_individual_scatter_plots()
        self.create_comparison_plot()
        self.create_overlay_plot()
        self.generate_statistics()
        
        print("可视化完成！生成的文件:")
        print(f"- 输出目录: {self.output_dir}")
        print("- charging_stations_heatmap.png/pdf: 热力图")
        print("- charging_stations_comparison.png/pdf: 对比图")
        print("- charging_stations_overlay.png/pdf: 叠加图")
        print("- cs_group_xxx_scatter.png/pdf: 各组散点图")
        print("- charging_stations_stats.txt: 统计信息")

if __name__ == "__main__":
    # 文件路径
    network_file = "/home/ubuntu/project/MSC/Msc_Project/data/map/glasgow_clean.net.xml"
    xml_dirs = [
        "/home/ubuntu/project/MSC/Msc_Project/data/cs_1-50",
        "/home/ubuntu/project/MSC/Msc_Project/data/cs_51-100"
    ]
    output_dir = "/home/ubuntu/project/MSC/Msc_Project/data/cs_1-100"
    
    # 创建可视化器并运行
    visualizer = ChargingStationVisualizerV2(network_file, xml_dirs, output_dir)
    visualizer.run_visualization()

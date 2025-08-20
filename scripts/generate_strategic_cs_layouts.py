#!/usr/bin/env python3
"""
生成具有战略差异的充电桩布局 (cs_group_051-100)
使用引导采样策略创建不同类型的布局：
- 中心集聚型：靠近城市中心
- 周边分散型：远离中心的分散布局
- 双中心型：围绕两个中心的布局
- 稀疏型：尽量分散的布局
- 密集型：高度聚集的布局
"""

import os
import json
import random
import argparse
import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

class StrategicLayoutGenerator:
    def __init__(self, net_file: str):
        self.net_file = net_file
        self.valid_edges = []
        self.edge_coords = {}
        self.city_center = None
        
        print("🔍 解析网络文件并提取边信息...")
        self._extract_edge_info()
        print(f"✅ 提取了 {len(self.valid_edges)} 条有效边")
        
    def _extract_edge_info(self):
        """从网络文件中提取边信息和坐标"""
        valid_type_whitelist = {
            "highway.living_street", "highway.primary", "highway.primary_link",
            "highway.residential", "highway.secondary", "highway.secondary_link",
            "highway.tertiary", "highway.tertiary_link", "highway.trunk",
            "highway.unclassified"
        }
        
        tree = ET.parse(self.net_file)
        root = tree.getroot()
        
        # 收集需要排除的边
        roundabout_edges = set()
        for elem in root.findall("roundabout"):
            edges = elem.get("edges", "")
            roundabout_edges.update(edges.split())
        
        deadend_junctions = set()
        for junction in root.findall("junction"):
            if junction.get("type") == "dead_end":
                deadend_junctions.add(junction.get("id"))
        
        edge_to_junction = {}
        for edge in root.findall("edge"):
            if edge.get("function") != "internal" and not edge.get("id", "").startswith(":"):
                to = edge.get("to")
                if to:
                    edge_to_junction[edge.get("id")] = to
        
        # 提取有效边及其坐标
        for edge in root.findall("edge"):
            edge_id = edge.get("id", "")
            edge_func = edge.get("function", "")
            edge_type = edge.get("type", "")
            
            # 过滤条件
            if (edge_func == "internal" or edge_id.startswith(":") or
                edge_id in roundabout_edges or
                edge_to_junction.get(edge_id) in deadend_junctions or
                edge_type not in valid_type_whitelist):
                continue
            
            # 检查lane是否合适
            for lane in edge.findall("lane"):
                length = float(lane.get("length", "0"))
                allow = lane.get("allow", "")
                shape = lane.get("shape", "")
                
                if length >= 10 and ("passenger" in allow or allow == "") and shape:
                    # 计算边的中心坐标
                    coords = []
                    for point_str in shape.split():
                        x, y = map(float, point_str.split(','))
                        coords.append((x, y))
                    
                    if coords:
                        # 使用边的中点作为代表坐标
                        center_x = np.mean([c[0] for c in coords])
                        center_y = np.mean([c[1] for c in coords])
                        
                        self.valid_edges.append(edge_id)
                        self.edge_coords[edge_id] = (center_x, center_y, length)
                        break
        
        # 计算城市中心（所有边的重心）
        if self.edge_coords:
            all_x = [coord[0] for coord in self.edge_coords.values()]
            all_y = [coord[1] for coord in self.edge_coords.values()]
            self.city_center = (np.mean(all_x), np.mean(all_y))
            print(f"🏙️ 城市中心坐标: ({self.city_center[0]:.1f}, {self.city_center[1]:.1f})")
    
    def _calculate_distance_to_center(self, edge_id: str) -> float:
        """计算边到城市中心的距离"""
        if edge_id not in self.edge_coords or not self.city_center:
            return float('inf')
        
        edge_coord = self.edge_coords[edge_id]
        return np.sqrt((edge_coord[0] - self.city_center[0])**2 + 
                      (edge_coord[1] - self.city_center[1])**2)
    
    def generate_center_clustered_layout(self, cs_count: int) -> List[Dict]:
        """生成中心集聚型布局 - 选择靠近城市中心的边"""
        print("🏙️ 生成中心集聚型布局...")
        
        # 计算所有边到中心的距离
        edge_distances = [(edge_id, self._calculate_distance_to_center(edge_id)) 
                         for edge_id in self.valid_edges]
        edge_distances.sort(key=lambda x: x[1])
        
        # 选择距离最近的60%的边
        center_ratio = 0.6
        center_edge_count = max(cs_count, int(len(edge_distances) * center_ratio))
        center_edges = [item[0] for item in edge_distances[:center_edge_count]]
        
        # 从中心边中随机选择
        selected_edges = random.sample(center_edges, min(cs_count, len(center_edges)))
        
        return self._generate_layout_data(selected_edges)
    
    def generate_peripheral_dispersed_layout(self, cs_count: int) -> List[Dict]:
        """生成周边分散型布局 - 选择远离城市中心的边"""
        print("🏞️ 生成周边分散型布局...")
        
        # 计算所有边到中心的距离
        edge_distances = [(edge_id, self._calculate_distance_to_center(edge_id)) 
                         for edge_id in self.valid_edges]
        edge_distances.sort(key=lambda x: x[1], reverse=True)
        
        # 选择距离最远的60%的边
        peripheral_ratio = 0.6
        peripheral_edge_count = max(cs_count, int(len(edge_distances) * peripheral_ratio))
        peripheral_edges = [item[0] for item in edge_distances[:peripheral_edge_count]]
        
        # 从周边边中随机选择
        selected_edges = random.sample(peripheral_edges, min(cs_count, len(peripheral_edges)))
        
        return self._generate_layout_data(selected_edges)
    
    def generate_dual_center_layout(self, cs_count: int) -> List[Dict]:
        """生成双中心型布局 - 围绕两个不同中心分布"""
        print("🎯 生成双中心型布局...")
        
        # 使用K-means找到两个聚类中心
        coords = np.array([self.edge_coords[edge_id][:2] for edge_id in self.valid_edges])
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords)
        
        # 获取两个聚类的边
        cluster_0_edges = [self.valid_edges[i] for i in range(len(self.valid_edges)) if cluster_labels[i] == 0]
        cluster_1_edges = [self.valid_edges[i] for i in range(len(self.valid_edges)) if cluster_labels[i] == 1]
        
        # 从每个聚类中选择一半
        half_count = cs_count // 2
        selected_edges = []
        
        if len(cluster_0_edges) >= half_count:
            selected_edges.extend(random.sample(cluster_0_edges, half_count))
        else:
            selected_edges.extend(cluster_0_edges)
        
        remaining_count = cs_count - len(selected_edges)
        if len(cluster_1_edges) >= remaining_count:
            selected_edges.extend(random.sample(cluster_1_edges, remaining_count))
        else:
            selected_edges.extend(cluster_1_edges)
            # 如果还不够，从剩余边中补充
            remaining_edges = [e for e in self.valid_edges if e not in selected_edges]
            if remaining_edges:
                needed = cs_count - len(selected_edges)
                selected_edges.extend(random.sample(remaining_edges, min(needed, len(remaining_edges))))
        
        return self._generate_layout_data(selected_edges)
    
    def generate_sparse_layout(self, cs_count: int) -> List[Dict]:
        """生成稀疏型布局 - 最大化边之间的距离（优化版本）"""
        print("📏 生成稀疏型布局...")
        
        if cs_count >= len(self.valid_edges):
            return self._generate_layout_data(self.valid_edges)
        
        # 优化策略：先预过滤减少候选集，然后使用网格采样
        print(f"   🔍 从 {len(self.valid_edges)} 条边中选择...")
        
        # 步骤1：预过滤 - 只使用部分边减少计算量
        if len(self.valid_edges) > 5000:
            # 随机采样1/3的边作为候选池
            sample_size = len(self.valid_edges) // 3
            candidate_pool = random.sample(self.valid_edges, sample_size)
            print(f"   📊 预过滤至 {len(candidate_pool)} 条候选边")
        else:
            candidate_pool = self.valid_edges
        
        # 步骤2：网格化分区策略
        coords = np.array([self.edge_coords[edge_id][:2] for edge_id in candidate_pool])
        
        # 计算边界
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        # 创建网格 - 根据cs_count动态调整网格大小
        grid_size = max(int(np.sqrt(cs_count * 2)), 5)  # 确保有足够的网格
        x_step = (x_max - x_min) / grid_size
        y_step = (y_max - y_min) / grid_size
        
        print(f"   🗺️ 使用 {grid_size}x{grid_size} 网格分区")
        
        # 将边分配到网格
        grid_edges = {}
        for i, edge_id in enumerate(candidate_pool):
            coord = coords[i]
            grid_x = int((coord[0] - x_min) / x_step) if x_step > 0 else 0
            grid_y = int((coord[1] - y_min) / y_step) if y_step > 0 else 0
            
            # 确保不越界
            grid_x = min(grid_x, grid_size - 1)
            grid_y = min(grid_y, grid_size - 1)
            
            grid_key = (grid_x, grid_y)
            if grid_key not in grid_edges:
                grid_edges[grid_key] = []
            grid_edges[grid_key].append(edge_id)
        
        # 步骤3：从每个网格中选择代表性边
        selected_edges = []
        
        # 从有边的网格中随机选择
        available_grids = list(grid_edges.keys())
        random.shuffle(available_grids)
        
        for grid_key in available_grids:
            if len(selected_edges) >= cs_count:
                break
            
            # 从当前网格随机选择一条边
            grid_edge_list = grid_edges[grid_key]
            selected_edge = random.choice(grid_edge_list)
            selected_edges.append(selected_edge)
        
        # 如果还不够，使用简化的贪心策略补充
        if len(selected_edges) < cs_count:
            print(f"   🔄 网格选择了 {len(selected_edges)} 条边，使用贪心策略补充...")
            remaining_needed = cs_count - len(selected_edges)
            remaining_pool = [e for e in candidate_pool if e not in selected_edges]
            
            # 简化贪心：每次从100个随机候选中选择最远的
            for _ in range(min(remaining_needed, len(remaining_pool))):
                if len(remaining_pool) == 0:
                    break
                
                # 限制搜索范围提高速度
                search_size = min(100, len(remaining_pool))
                search_candidates = random.sample(remaining_pool, search_size)
                
                best_edge = None
                max_min_distance = -1
                
                for candidate_edge in search_candidates:
                    candidate_coord = np.array(self.edge_coords[candidate_edge][:2])
                    
                    # 只计算到最近几个已选边的距离
                    min_distance = float('inf')
                    check_count = min(5, len(selected_edges))  # 只检查最近的5个
                    
                    for selected_edge in selected_edges[-check_count:]:
                        selected_coord = np.array(self.edge_coords[selected_edge][:2])
                        distance = np.linalg.norm(candidate_coord - selected_coord)
                        min_distance = min(min_distance, distance)
                    
                    if min_distance > max_min_distance:
                        max_min_distance = min_distance
                        best_edge = candidate_edge
                
                if best_edge:
                    selected_edges.append(best_edge)
                    remaining_pool.remove(best_edge)
        
        print(f"   ✅ 选择了 {len(selected_edges)} 条边用于稀疏布局")
        return self._generate_layout_data(selected_edges)
    
    def generate_dense_layout(self, cs_count: int) -> List[Dict]:
        """生成密集型布局 - 选择相互邻近的边（优化版本）"""
        print("🔗 生成密集型布局...")
        
        if cs_count >= len(self.valid_edges):
            return self._generate_layout_data(self.valid_edges)
        
        # 优化策略：使用KMeans找到密集区域，然后在区域内选择
        print(f"   🔍 从 {len(self.valid_edges)} 条边中选择...")
        
        # 步骤1：预过滤减少计算量
        if len(self.valid_edges) > 5000:
            sample_size = min(5000, len(self.valid_edges))
            candidate_pool = random.sample(self.valid_edges, sample_size)
            print(f"   📊 预过滤至 {len(candidate_pool)} 条候选边")
        else:
            candidate_pool = self.valid_edges
        
        # 步骤2：使用聚类找到密集区域
        coords = np.array([self.edge_coords[edge_id][:2] for edge_id in candidate_pool])
        
        # 使用较多的聚类中心
        n_clusters = max(5, cs_count // 20)  # 每个聚类约20个充电桩
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords)
        
        print(f"   🎯 使用 {n_clusters} 个聚类中心")
        
        # 步骤3：选择最大的聚类作为密集区域
        cluster_sizes = {}
        for i, label in enumerate(cluster_labels):
            if label not in cluster_sizes:
                cluster_sizes[label] = []
            cluster_sizes[label].append(i)
        
        # 按聚类大小排序
        sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: len(x[1]), reverse=True)
        
        selected_edges = []
        
        # 从最大的聚类开始选择
        for cluster_id, edge_indices in sorted_clusters:
            if len(selected_edges) >= cs_count:
                break
            
            cluster_edges = [candidate_pool[i] for i in edge_indices]
            
            # 从当前聚类中选择边
            needed = min(len(cluster_edges), cs_count - len(selected_edges))
            
            if needed > 0:
                # 在聚类内随机选择
                selected_cluster_edges = random.sample(cluster_edges, needed)
                selected_edges.extend(selected_cluster_edges)
                
                print(f"   📍 从聚类 {cluster_id} 选择了 {needed} 条边")
        
        # 如果还不够，随机补充
        if len(selected_edges) < cs_count:
            remaining_edges = [e for e in candidate_pool if e not in selected_edges]
            needed = cs_count - len(selected_edges)
            if remaining_edges:
                additional_edges = random.sample(remaining_edges, min(needed, len(remaining_edges)))
                selected_edges.extend(additional_edges)
                print(f"   🔄 随机补充了 {len(additional_edges)} 条边")
        
        print(f"   ✅ 选择了 {len(selected_edges)} 条边用于密集布局")
        return self._generate_layout_data(selected_edges)
    
    def _generate_layout_data(self, selected_edges: List[str]) -> List[Dict]:
        """为选定的边生成充电桩位置数据"""
        layout_data = []
        
        for edge_id in selected_edges:
            if edge_id in self.edge_coords:
                length = self.edge_coords[edge_id][2]
                max_pos = max(length - 5.0, 0.0)
                
                if max_pos > 0:
                    pos = round(random.uniform(0.0, max_pos), 1)
                    layout_data.append({
                        "edge_id": edge_id,
                        "pos": pos
                    })
        
        return layout_data
    
    def generate_all_layouts(self, cs_count: int = 215, start_id: int = 51, layout_config: Dict = None) -> Dict:
        """生成指定类型的布局
        Args:
            cs_count: 每个布局的充电桩数量
            start_id: 起始布局ID
            layout_config: 布局配置，格式为 {"layout_type": count, ...}
        """
        if layout_config is None:
            # 默认配置：所有类型各10个
            layout_config = {
                "center_clustered": 10,
                "peripheral_dispersed": 10,
                "dual_center": 10,
                "sparse": 10,
                "dense": 10
            }
        
        layout_registry = {}
        layout_counter = start_id
        
        layout_type_mapping = {
            "center_clustered": ("center_clustered", "中心集聚型", self.generate_center_clustered_layout),
            "peripheral_dispersed": ("peripheral_dispersed", "周边分散型", self.generate_peripheral_dispersed_layout),
            "dual_center": ("dual_center", "双中心型", self.generate_dual_center_layout),
            "sparse": ("sparse", "稀疏型", self.generate_sparse_layout),
            "dense": ("dense", "密集型", self.generate_dense_layout)
        }
        
        for layout_type, count in layout_config.items():
            if layout_type not in layout_type_mapping:
                print(f"⚠️ 未知的布局类型: {layout_type}")
                continue
                
            layout_key, type_name, generator_func = layout_type_mapping[layout_type]
            print(f"\n🎨 生成 {type_name} 布局 ({count}个)...")
            
            for i in range(count):
                layout_id = f"cs_group_{layout_counter:03d}"
                print(f"   🎯 生成 {layout_id} ({i+1}/{count})")
                
                # 为每个布局设置不同的随机种子，确保多样性
                random.seed(layout_counter * 100 + i)
                np.random.seed(layout_counter * 100 + i)
                
                layout_data = generator_func(cs_count)
                
                layout_registry[layout_id] = {
                    "layout_type": layout_key,
                    "type_name": type_name,
                    "cs_count": len(layout_data),
                    "data": layout_data
                }
                
                layout_counter += 1
        
        print(f"\n✅ 共生成 {len(layout_registry)} 个战略布局")
        return layout_registry

def main():
    parser = argparse.ArgumentParser(description='生成战略性充电桩布局')
    parser.add_argument('--net_file', type=str,
                       default="/home/ubuntu/project/MSC/Msc_Project/data/map/glasgow_clean.net.xml",
                       help='网络文件路径')
    parser.add_argument('--output_dir', type=str,
                       default="/home/ubuntu/project/MSC/Msc_Project/data/cs_51-100",
                       help='输出目录路径')
    parser.add_argument('--start_id', type=int, default=51,
                       help='起始布局ID (默认: 51)')
    parser.add_argument('--end_id', type=int, default=100,
                       help='结束布局ID (默认: 100)')
    parser.add_argument('--cs_count', type=int, default=215,
                       help='每个布局的充电桩数量 (默认: 215)')
    parser.add_argument('--center_count', type=int, default=10,
                       help='中心集聚型布局数量 (默认: 10)')
    parser.add_argument('--peripheral_count', type=int, default=10,
                       help='周边分散型布局数量 (默认: 10)')
    parser.add_argument('--dual_count', type=int, default=10,
                       help='双中心型布局数量 (默认: 10)')
    parser.add_argument('--sparse_count', type=int, default=10,
                       help='稀疏型布局数量 (默认: 10)')
    parser.add_argument('--dense_count', type=int, default=10,
                       help='密集型布局数量 (默认: 10)')
    
    args = parser.parse_args()
    
    # 根据start_id和end_id推断输出文件名
    if args.start_id == 51 and args.end_id == 70:
        # 特殊情况：只生成51-70
        output_file_suffix = "51-70"
        default_config = {
            "center_clustered": args.center_count,
            "peripheral_dispersed": args.peripheral_count
        }
    elif args.start_id == 51 and args.end_id == 100:
        # 默认情况：生成51-100
        output_file_suffix = "51-100"
        default_config = {
            "center_clustered": args.center_count,
            "peripheral_dispersed": args.peripheral_count,
            "dual_center": args.dual_count,
            "sparse": args.sparse_count,
            "dense": args.dense_count
        }
    else:
        # 自定义范围
        output_file_suffix = f"{args.start_id}-{args.end_id}"
        default_config = {
            "center_clustered": args.center_count,
            "peripheral_dispersed": args.peripheral_count,
            "dual_center": args.dual_count,
            "sparse": args.sparse_count,
            "dense": args.dense_count
        }
    
    print(f"🚀 开始生成战略性充电桩布局 (cs_group_{args.start_id:03d}-{args.end_id:03d})")
    
    # 设置路径
    output_dir = args.output_dir
    output_file = os.path.join(output_dir, f"cs_candidates_{output_file_suffix}.json")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 网络文件: {args.net_file}")
    print(f"📁 输出目录: {output_dir}")
    print(f"💾 输出文件: {output_file}")
    
    # 检查网络文件是否存在
    if not os.path.exists(args.net_file):
        print(f"❌ 网络文件不存在: {args.net_file}")
        parser.print_help()
        print("\n💡 使用示例:")
        print("   # 生成cs_group_051-070 (10个中心集聚型 + 10个周边分散型)")
        print("   python scripts/generate_strategic_cs_layouts.py --start_id 51 --end_id 70 --output_dir data/cs_51-70 --center_count 10 --peripheral_count 10 --dual_count 0 --sparse_count 0 --dense_count 0")
        print("\n   # 生成cs_group_051-100 (所有类型各10个)")
        print("   python scripts/generate_strategic_cs_layouts.py --start_id 51 --end_id 100 --output_dir data/cs_51-100")
        return 1
    
    try:
        # 创建生成器
        generator = StrategicLayoutGenerator(args.net_file)
        
        # 生成指定布局
        layout_registry = generator.generate_all_layouts(
            cs_count=args.cs_count, 
            start_id=args.start_id, 
            layout_config=default_config
        )
        
        # 转换为候选格式（只保留data部分）
        candidates_data = {}
        for layout_id, layout_info in layout_registry.items():
            candidates_data[layout_id] = layout_info["data"]
        
        # 保存候选数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(candidates_data, f, indent=2, ensure_ascii=False)
        
        # 保存完整的布局信息（包含类型信息）
        registry_file = os.path.join(output_dir, f"layout_registry_{output_file_suffix}.json")
        with open(registry_file, 'w', encoding='utf-8') as f:
            json.dump(layout_registry, f, indent=2, ensure_ascii=False)
        
        print(f"\n🎉 战略布局生成完成！")
        print(f"💾 候选数据保存至: {output_file}")
        print(f"📋 完整注册表: {registry_file}")
        
        # 显示布局类型统计
        print(f"\n📊 布局类型分布:")
        type_counts = {}
        for layout_info in layout_registry.values():
            type_name = layout_info["type_name"]
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        for type_name, count in type_counts.items():
            print(f"   - {type_name}: {count} 个")
        
        start_layout = f"cs_group_{args.start_id:03d}"
        end_layout = f"cs_group_{args.start_id + len(layout_registry) - 1:03d}"
        print(f"\n🏷️ 布局ID范围: {start_layout} - {end_layout}")
        
        return 0
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())

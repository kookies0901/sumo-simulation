import os
import json
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from collections import defaultdict
from math import sqrt
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from sklearn.cluster import DBSCAN

def extract_layout_features_from_json(cs_group_data, net_xml_path):
    """从JSON数据中提取布局特征"""
    # Load edge information from net.xml
    edge_shapes = {}
    net_tree = ET.parse(net_xml_path)
    
    # 提取边的几何信息
    for edge in net_tree.findall(".//edge"):
        edge_id = edge.get("id")
        for lane in edge.findall("lane"):
            lane_id = lane.get("id")
            shape = lane.get("shape")
            if shape and edge_id:
                # 解析shape字符串，获取坐标点列表
                points = []
                for point_str in shape.split():
                    x, y = map(float, point_str.split(','))
                    points.append((x, y))
                edge_shapes[edge_id] = points
                # 使用第一个lane代表这条edge
                break
    
    # 计算充电桩的实际坐标
    cs_coords = []
    for cs_data in cs_group_data:
        edge_id = cs_data["edge_id"]
        pos = cs_data["pos"]
        
        if edge_id in edge_shapes:
            # 根据position在edge上插值得到坐标
            coord = interpolate_position_on_edge(edge_shapes[edge_id], pos)
            if coord:
                cs_coords.append(coord)

    # 如果找不到任何坐标就返回空
    if not cs_coords:
        raise ValueError(f"No valid charging station coordinates found")
    
    # 调用原有的特征计算逻辑
    return calculate_features_from_coords(cs_coords, net_xml_path)

def interpolate_position_on_edge(edge_points, position):
    """在边上根据位置插值得到坐标"""
    try:
        if len(edge_points) < 2:
            return edge_points[0] if edge_points else None
        
        # 计算边的总长度
        total_length = 0
        segment_lengths = []
        for i in range(len(edge_points) - 1):
            p1 = np.array(edge_points[i])
            p2 = np.array(edge_points[i + 1])
            length = np.linalg.norm(p2 - p1)
            segment_lengths.append(length)
            total_length += length
        
        if total_length == 0:
            return edge_points[0]
        
        # 如果position超出边长，返回端点
        if position <= 0:
            return edge_points[0]
        if position >= total_length:
            return edge_points[-1]
        
        # 找到position对应的线段
        current_length = 0
        for i, seg_length in enumerate(segment_lengths):
            if current_length + seg_length >= position:
                # 在当前线段上插值
                t = (position - current_length) / seg_length if seg_length > 0 else 0
                p1 = np.array(edge_points[i])
                p2 = np.array(edge_points[i + 1])
                interpolated = p1 + t * (p2 - p1)
                return tuple(interpolated)
            current_length += seg_length
        
        # 默认返回最后一个点
        return edge_points[-1]
        
    except Exception as e:
        print(f"Warning: 插值计算失败: {e}")
        return edge_points[0] if edge_points else None

def calculate_features_from_coords(cs_coords, net_xml_path):
    """从充电桩坐标计算特征"""
    # 转成 numpy array 处理
    coords = np.array(cs_coords)
    cs_count = len(coords)

    # 中心点
    center_x, center_y = coords.mean(axis=0)

    # 计算特征
    dists_to_center = np.linalg.norm(coords - [center_x, center_y], axis=1)
    avg_dist_to_center = dists_to_center.mean()

    if cs_count > 1:
        pairwise_dists = pdist(coords)
        avg_nn = pairwise_dists.mean()
        std_nn = pairwise_dists.std()
        min_nn = pairwise_dists.min()
        
        # 新增特征：最大两两距离
        max_pairwise_distance = pairwise_dists.max()
        
        # 新增特征：计算密度标准差
        # 将区域划分为网格，计算每个网格的桩数密度
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        
        # 计算边界
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # 划分网格（使用10x10网格）
        grid_size = 10
        x_bins = np.linspace(x_min, x_max, grid_size + 1)
        y_bins = np.linspace(y_min, y_max, grid_size + 1)
        
        # 计算每个网格的桩数
        grid_counts = np.zeros((grid_size, grid_size))
        for i in range(cs_count):
            x_idx = np.digitize(x_coords[i], x_bins) - 1
            y_idx = np.digitize(y_coords[i], y_bins) - 1
            if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size:
                grid_counts[x_idx, y_idx] += 1
        
        # 计算每个网格的面积（平方公里）
        grid_area_km2 = ((x_max - x_min) / grid_size) * ((y_max - y_min) / grid_size) / 1000000
        
        # 计算密度（桩数/平方公里）
        densities = grid_counts.flatten() / grid_area_km2
        # 过滤掉没有桩的网格
        densities = densities[densities > 0]
        
        if len(densities) > 0:
            cs_density_std = np.std(densities)
        else:
            cs_density_std = 0.0
            
        # 新增特征：聚类簇数（使用DBSCAN算法）
        cluster_count = calculate_dbscan_cluster_count(coords)
            
    else:
        avg_nn = std_nn = min_nn = 0.0
        max_pairwise_distance = 0.0
        cs_density_std = 0.0
        cluster_count = 0 if cs_count == 0 else 1

    # 新增特征：空间覆盖与均衡度指标
    coverage_ratio, max_gap_distance, gini_coefficient = calculate_enhanced_coverage_features(
        cs_coords, net_xml_path
    )
    
    # 新增特征：网络位置指标（使用NetworkX）
    avg_betweenness_centrality = calculate_network_centrality(cs_coords, {}, net_xml_path)

    # 构建输出字典
    features = {
        "cs_count": cs_count,
        "avg_dist_to_center": avg_dist_to_center,
        "avg_nearest_neighbor": avg_nn,
        "std_nearest_neighbor": std_nn,
        "min_distance": min_nn,
        "max_pairwise_distance": max_pairwise_distance,
        "cs_density_std": cs_density_std,
        # 新增特征
        "cluster_count": cluster_count,
        "coverage_ratio": coverage_ratio,
        "max_gap_distance": max_gap_distance,
        "gini_coefficient": gini_coefficient,
        "avg_betweenness_centrality": avg_betweenness_centrality
    }

    return features

def extract_layout_features(cs_xml_path, net_xml_path):
    """从XML文件中提取布局特征（保持向后兼容）"""
    # Load all lane positions from net.xml
    lane_coords = {}
    net_tree = ET.parse(net_xml_path)
    for lane in net_tree.findall(".//lane"):
        lane_id = lane.get("id")
        shape = lane.get("shape")
        if shape:
            # 取shape第一个点作为代表坐标
            x, y = map(float, shape.split()[0].split(','))
            lane_coords[lane_id] = (x, y)

    # Load charging stations from cs xml
    cs_coords = []
    cs_tree = ET.parse(cs_xml_path)
    for cs in cs_tree.findall(".//chargingStation"):
        lane_id = cs.get("lane")
        if lane_id in lane_coords:
            cs_coords.append(lane_coords[lane_id])

    # 如果找不到任何坐标就返回空
    if not cs_coords:
        raise ValueError(f"No valid chargingStation coordinates found in {cs_xml_path}")

    return calculate_features_from_coords(cs_coords, net_xml_path)

def calculate_dbscan_cluster_count(coords):
    """使用DBSCAN算法进行聚类分析"""
    try:
        if len(coords) <= 1:
            return 0 if len(coords) == 0 else 1
        
        # 使用DBSCAN聚类算法
        # eps: 邻域半径（500米），min_samples: 最小样本数
        eps = 500  # 500米邻域半径
        min_samples = 2  # 最少2个点形成一个簇
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(coords)
        
        # 计算聚类数量（排除噪声点，标签为-1）
        unique_labels = set(cluster_labels)
        cluster_count = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        # 如果没有形成聚类，则每个点都是独立的聚类
        if cluster_count == 0:
            cluster_count = len(coords)
        
        return cluster_count
        
    except Exception as e:
        print(f"Warning: DBSCAN聚类计算失败: {e}")
        # 回退到简单方法
        return calculate_simple_cluster_count(coords)

def calculate_simple_cluster_count(coords):
    """简化版本的聚类计数，基于距离阈值（作为DBSCAN的备用方案）"""
    try:
        if len(coords) <= 1:
            return len(coords)
        
        # 使用简单的距离阈值方法
        threshold = 500  # 500米阈值，与DBSCAN保持一致
        clusters = []
        visited = set()
        
        for i in range(len(coords)):
            if i in visited:
                continue
                
            # 开始新聚类
            cluster = [i]
            visited.add(i)
            
            # 查找所有在阈值内的点
            for j in range(i + 1, len(coords)):
                if j not in visited:
                    dist = np.linalg.norm(coords[i] - coords[j])
                    if dist <= threshold:
                        cluster.append(j)
                        visited.add(j)
            
            clusters.append(cluster)
        
        return len(clusters)
        
    except:
        return 0

# 全局缓存采样点
_cached_sample_coords = None

def calculate_enhanced_coverage_features(cs_coords, net_xml_path):
    """增强版的覆盖特征计算，直接使用网络文件（使用缓存）"""
    global _cached_sample_coords
    
    try:
        # 使用缓存的采样点
        if _cached_sample_coords is None:
            print("🔧 首次采样道路点...")
            _cached_sample_coords = sample_road_points_from_network(net_xml_path)
            print("✅ 道路点采样完成，后续场景将使用缓存")
        
        edge_sample_coords = _cached_sample_coords
        
        if not edge_sample_coords:
            return 0.0, 0.0, 0.0
        
        # 计算每个采样点到最近充电桩的距离
        distances_to_cs = []
        for edge_coord in edge_sample_coords:
            # 计算到所有充电桩的距离，取最小值
            dists = [np.linalg.norm(np.array(edge_coord) - np.array(cs_coord)) 
                    for cs_coord in cs_coords]
            min_dist = min(dists)
            distances_to_cs.append(min_dist)
        
        distances_to_cs = np.array(distances_to_cs)
        
        # 1. Coverage ratio (500m内可到达充电桩的路段比例)
        coverage_ratio = np.mean(distances_to_cs <= 500)
        
        # 2. Max gap distance (最大服务空白距离)
        max_gap_distance = np.max(distances_to_cs)
        
        # 3. Gini coefficient for service accessibility
        gini_coefficient = calculate_gini_coefficient(distances_to_cs)
        
        return coverage_ratio, max_gap_distance, gini_coefficient
        
    except Exception as e:
        print(f"Warning: 计算增强覆盖特征时出错: {e}")
        return 0.0, 0.0, 0.0

def sample_road_points_from_network(net_xml_path, sample_ratio=0.1):
    """从网络文件中采样道路点"""
    try:
        tree = ET.parse(net_xml_path)
        root = tree.getroot()
        
        sample_coords = []
        
        # 遍历所有边，采样坐标点
        for edge in root.findall(".//edge"):
            edge_id = edge.get("id")
            # 跳过内部边
            if edge_id and edge_id.startswith(":"):
                continue
                
            for lane in edge.findall("lane"):
                shape = lane.get("shape")
                if shape:
                    # 解析shape并采样
                    points = []
                    for point_str in shape.split():
                        x, y = map(float, point_str.split(','))
                        points.append((x, y))
                    
                    # 采样一定比例的点
                    num_samples = max(1, int(len(points) * sample_ratio))
                    step = max(1, len(points) // num_samples)
                    
                    for i in range(0, len(points), step):
                        sample_coords.append(points[i])
                    
                    # 只处理第一个lane
                    break
        
        print(f"从网络中采样了 {len(sample_coords)} 个道路点")
        return sample_coords
        
    except Exception as e:
        print(f"Warning: 从网络采样道路点失败: {e}")
        return []

def calculate_simple_coverage_features(cs_coords, lane_coords):
    """简化版本的覆盖特征计算"""
    try:
        # 抽样策略：选择10%的代表性路段作为虚拟需求点
        all_edges = list(lane_coords.keys())
        sample_size = max(1, int(len(all_edges) * 0.1))  # 10%抽样
        
        # 使用固定的随机种子确保结果可重现
        np.random.seed(42)
        sampled_edges = np.random.choice(all_edges, sample_size, replace=False)
        
        # 计算每个采样路段到最近充电桩的距离
        distances_to_cs = []
        for edge_id in sampled_edges:
            edge_coord = lane_coords[edge_id]
            # 计算到所有充电桩的距离，取最小值
            dists = [np.linalg.norm(np.array(edge_coord) - np.array(cs_coord)) 
                    for cs_coord in cs_coords]
            min_dist = min(dists)
            distances_to_cs.append(min_dist)
        
        distances_to_cs = np.array(distances_to_cs)
        
        # 1. Coverage ratio (500m内可到达充电桩的路段比例)
        coverage_ratio = np.mean(distances_to_cs <= 500)
        
        # 2. Max gap distance (最大服务空白距离)
        max_gap_distance = np.max(distances_to_cs)
        
        # 3. Gini coefficient for service accessibility
        gini_coefficient = calculate_gini_coefficient(distances_to_cs)
        
        return coverage_ratio, max_gap_distance, gini_coefficient
        
    except Exception as e:
        print(f"Warning: 计算覆盖特征时出错: {e}")
        return 0.0, 0.0, 0.0

def calculate_gini_coefficient(values):
    """计算基尼系数"""
    try:
        if len(values) == 0:
            return 0.0
        
        # 排序
        sorted_values = np.sort(values)
        n = len(sorted_values)
        
        # 计算基尼系数
        cumsum = np.cumsum(sorted_values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0.0
        
    except:
        return 0.0

# 全局缓存网络图和中心性
_cached_network = None
_cached_centrality = None

def calculate_network_centrality(cs_coords, lane_coords, net_xml_path):
    """计算网络中心性指标（使用缓存优化性能）"""
    global _cached_network, _cached_centrality
    
    try:
        # 使用缓存的网络图
        if _cached_network is None:
            print("🔧 首次构建路网图（较慢）...")
            _cached_network = build_road_network(net_xml_path)
            if _cached_network is None:
                print("Warning: 无法构建有效的路网图")
                return 0.0
        
        G = _cached_network
        
        # 使用缓存的中心性计算
        if _cached_centrality is None:
            print("🔧 首次计算中心性（较慢）...")
            try:
                # 使用合理的采样数量保证精度
                if len(G.nodes()) > 5000:
                    # 采样1000个节点保证计算精度
                    _cached_centrality = nx.betweenness_centrality(G, k=min(1000, len(G.nodes())), normalized=True)
                else:
                    _cached_centrality = nx.betweenness_centrality(G, normalized=True)
                print("✅ 中心性计算完成，后续场景将使用缓存")
            except Exception as e:
                print(f"Warning: 中心性计算失败，使用简化方法: {e}")
                return calculate_simple_centrality(cs_coords, G)
        
        centrality = _cached_centrality
        
        # 为充电桩找到最近的路网节点
        cs_nodes = []
        for cs_coord in cs_coords:
            closest_node = find_closest_network_node(G, cs_coord)
            if closest_node is not None:
                cs_nodes.append(closest_node)
        
        if not cs_nodes:
            return 0.0
        
        # 计算充电桩位置的平均介数中心性
        cs_centralities = [centrality.get(node, 0.0) for node in cs_nodes]
        avg_centrality = np.mean(cs_centralities) if cs_centralities else 0.0
        
        return avg_centrality
    
    except Exception as e:
        print(f"Warning: 网络中心性计算失败: {e}")
        return 0.0

def calculate_simple_centrality(cs_coords, G):
    """简化的中心性计算（备用方案）"""
    try:
        # 简单基于度中心性
        degree_centrality = nx.degree_centrality(G)
        
        cs_nodes = []
        for cs_coord in cs_coords:
            closest_node = find_closest_network_node(G, cs_coord)
            if closest_node is not None:
                cs_nodes.append(closest_node)
        
        if not cs_nodes:
            return 0.0
        
        cs_centralities = [degree_centrality.get(node, 0.0) for node in cs_nodes]
        return np.mean(cs_centralities) if cs_centralities else 0.0
        
    except:
        return 0.0

def build_road_network(net_xml_path):
    """从net.xml构建路网图"""
    try:
        G = nx.Graph()
        tree = ET.parse(net_xml_path)
        root = tree.getroot()
        
        # 添加节点
        for junction in root.findall(".//junction"):
            junction_id = junction.get("id")
            x = float(junction.get("x", 0))
            y = float(junction.get("y", 0))
            G.add_node(junction_id, pos=(x, y))
        
        # 添加边（基于edge连接）
        for edge in root.findall(".//edge"):
            edge_id = edge.get("id")
            from_node = edge.get("from")
            to_node = edge.get("to")
            
            # 跳过内部边
            if edge_id and edge_id.startswith(":"):
                continue
                
            if from_node and to_node and from_node in G.nodes() and to_node in G.nodes():
                # 计算边的长度作为权重
                from_pos = G.nodes[from_node]["pos"]
                to_pos = G.nodes[to_node]["pos"]
                length = np.linalg.norm(np.array(from_pos) - np.array(to_pos))
                G.add_edge(from_node, to_node, weight=length, edge_id=edge_id)
        
        print(f"路网图构建完成: {len(G.nodes())} 个节点, {len(G.edges())} 条边")
        return G
        
    except Exception as e:
        print(f"Warning: 路网构建失败: {e}")
        return None

def find_closest_network_node(G, target_coord):
    """找到距离目标坐标最近的网络节点"""
    try:
        min_distance = float('inf')
        closest_node = None
        
        target_pos = np.array(target_coord)
        
        for node_id, data in G.nodes(data=True):
            if "pos" in data:
                node_pos = np.array(data["pos"])
                distance = np.linalg.norm(target_pos - node_pos)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_node = node_id
        
        return closest_node
    
    except Exception as e:
        print(f"Warning: 寻找最近节点失败: {e}")
        return None

def extract_all_layout_features_from_json(json_file_path, net_xml_path, output_dir=None):
    """从JSON文件批量提取所有充电桩布局的特征"""
    if output_dir is None:
        output_dir = os.path.dirname(json_file_path)
    
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            cs_data = json.load(f)
        
        print(f"📊 加载了 {len(cs_data)} 个充电桩布局组")
        
        # 检查已完成的文件，支持断点续传
        completed_groups = set()
        final_output_file = os.path.join(output_dir, "all_layout_features.csv")
        if os.path.exists(final_output_file):
            try:
                existing_df = pd.read_csv(final_output_file)
                completed_groups = set(existing_df['layout_id'].tolist())
                print(f"🔄 发现已完成的布局组: {len(completed_groups)} 个")
            except:
                pass
        
        all_features = []
        success_count = 0
        total_count = len(cs_data)
        skipped_count = 0
        
        for idx, (group_id, group_data) in enumerate(cs_data.items(), 1):
            # 检查是否已完成
            if group_id in completed_groups:
                skipped_count += 1
                print(f"\n[{idx}/{total_count}] ⏭️ 跳过已完成: {group_id}")
                continue
                
            print(f"\n[{idx}/{total_count}] 处理布局组: {group_id}")
            
            try:
                features = extract_layout_features_from_json(group_data, net_xml_path)
                features["layout_id"] = group_id
                all_features.append(features)
                success_count += 1
                print(f"✅ 提取特征完成: {group_id} ({len(group_data)} 个充电桩)")
                
                # 立即保存单个结果
                single_df = pd.DataFrame([features])
                single_output_file = os.path.join(output_dir, f"{group_id}_layout_features.csv")
                single_df.to_csv(single_output_file, index=False)
                print(f"💾 单个文件已保存: {single_output_file}")
                
                # 立即更新汇总文件
                if os.path.exists(final_output_file):
                    # 读取现有文件并追加
                    existing_df = pd.read_csv(final_output_file)
                    updated_df = pd.concat([existing_df, single_df], ignore_index=True)
                    updated_df.to_csv(final_output_file, index=False)
                else:
                    # 创建新文件
                    single_df.to_csv(final_output_file, index=False)
                print(f"💾 汇总文件已更新: {final_output_file}")
                
                # 每处理5个就备份一次
                if success_count % 5 == 0:
                    backup_file = os.path.join(output_dir, f"all_layout_features_backup_{success_count}.csv")
                    if os.path.exists(final_output_file):
                        import shutil
                        shutil.copy2(final_output_file, backup_file)
                        print(f"🔄 备份文件: {backup_file}")
                
            except Exception as e:
                print(f"❌ 提取特征失败 {group_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 保存到CSV文件
        if all_features:
            # 保存所有特征到一个文件
            df = pd.DataFrame(all_features)
            output_file = os.path.join(output_dir, "all_layout_features.csv")
            df.to_csv(output_file, index=False)
            print(f"\n✅ 所有特征保存到: {output_file}")
            
            # 同时为每个布局生成单独的特征文件
            for features in all_features:
                layout_id = features["layout_id"]
                single_df = pd.DataFrame([features])
                single_output_file = os.path.join(output_dir, f"{layout_id}_layout_features.csv")
                single_df.to_csv(single_output_file, index=False)
            
            print(f"🎉 批量处理完成！")
            print(f"✅ 成功处理: {success_count} 个")
            print(f"⏭️ 跳过已完成: {skipped_count} 个") 
            print(f"📊 总计: {success_count + skipped_count}/{len(cs_data)} 个布局组")
            
        return all_features
        
    except Exception as e:
        print(f"❌ 处理JSON文件失败: {e}")
        return []

def extract_all_layout_features(cs_dir, net_xml_path, output_dir=None):
    """批量提取所有充电桩布局的特征（XML文件版本，保持向后兼容）"""
    if output_dir is None:
        output_dir = cs_dir
    
    # 获取所有cs_group_*.xml文件
    cs_files = []
    for file in os.listdir(cs_dir):
        if file.startswith("cs_group_") and file.endswith(".xml"):
            cs_files.append(file)
    
    cs_files.sort()  # 按文件名排序
    
    all_features = []
    
    for cs_file in cs_files:
        cs_xml_path = os.path.join(cs_dir, cs_file)
        layout_id = cs_file.replace(".xml", "")
        
        try:
            features = extract_layout_features(cs_xml_path, net_xml_path)
            features["layout_id"] = layout_id
            all_features.append(features)
            print(f"✅ 提取特征完成: {layout_id}")
        except Exception as e:
            print(f"❌ 提取特征失败 {layout_id}: {e}")
            continue
    
    # 保存到CSV文件
    if all_features:
        df = pd.DataFrame(all_features)
        output_file = os.path.join(output_dir, "all_layout_features.csv")
        df.to_csv(output_file, index=False)
        print(f"✅ 所有特征保存到: {output_file}")
        
        # 同时为每个布局生成单独的特征文件
        for features in all_features:
            layout_id = features["layout_id"]
            single_df = pd.DataFrame([features])
            single_output_file = os.path.join(output_dir, f"{layout_id}_layout_features.csv")
            single_df.to_csv(single_output_file, index=False)
    
    return all_features

def main():
    parser = argparse.ArgumentParser(description='提取充电桩布局特征')
    parser.add_argument('--layout_file', type=str, 
                       help='布局文件路径 (JSON格式，如 cs_candidates.json)')
    parser.add_argument('--net_file', type=str,
                       default='data/map/glasgow_clean.net.xml',
                       help='网络文件路径 (默认: data/map/glasgow_clean.net.xml)')
    parser.add_argument('--output_dir', type=str,
                       help='输出目录路径 (默认: 布局文件所在目录)')
    
    args = parser.parse_args()
    
    # 如果没有提供布局文件参数，使用默认值并显示帮助
    if not args.layout_file:
        print("🚀 开始处理充电桩布局特征提取")
        print("💡 未指定布局文件，使用默认配置")
        
        # 默认配置
        json_file_path = "/home/ubuntu/project/MSC/Msc_Project/data/cs/cs_candidates.json"
        net_xml_path = "/home/ubuntu/project/MSC/Msc_Project/data/map/glasgow_clean.net.xml"
        output_dir = "/home/ubuntu/project/MSC/Msc_Project/data/cs"
    else:
        # 使用命令行参数
        json_file_path = args.layout_file
        net_xml_path = args.net_file
        output_dir = args.output_dir if args.output_dir else os.path.dirname(args.layout_file)
        
        print("🚀 开始处理充电桩布局特征提取")
        print(f"📊 使用指定的布局文件: {json_file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(json_file_path):
        print(f"❌ JSON文件不存在: {json_file_path}")
        parser.print_help()
        print("\n💡 使用示例:")
        print("   # 使用默认配置")
        print("   python scripts/extract_layout_features.py")
        print("\n   # 指定布局文件")
        print("   python scripts/extract_layout_features.py --layout_file data/cs_51-100/cs_candidates_51-100.json")
        print("\n   # 指定布局文件和网络文件")
        print("   python scripts/extract_layout_features.py --layout_file data/cs_51-100/cs_candidates_51-100.json --net_file data/map/glasgow_clean.net.xml")
        print("\n   # 指定输出目录")
        print("   python scripts/extract_layout_features.py --layout_file data/cs_51-100/cs_candidates_51-100.json --output_dir output/features")
        exit(1)
    
    if not os.path.exists(net_xml_path):
        print(f"❌ 网络文件不存在: {net_xml_path}")
        exit(1)
    
    print(f"📊 JSON文件: {json_file_path}")
    print(f"🗺️ 网络文件: {net_xml_path}")
    print(f"💾 输出目录: {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 执行批量特征提取
    try:
        all_features = extract_all_layout_features_from_json(
            json_file_path=json_file_path,
            net_xml_path=net_xml_path,
            output_dir=output_dir
        )
        
        if all_features:
            print(f"\n🎉 特征提取完成！共处理 {len(all_features)} 个布局组")
            
            # 显示第一个样例的特征
            if len(all_features) > 0:
                print(f"\n📋 样例特征 ({all_features[0]['layout_id']}):")
                for k, v in all_features[0].items():
                    if k != 'layout_id':
                        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        else:
            print("❌ 没有成功提取任何特征")
            
    except Exception as e:
        print(f"❌ 批量处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


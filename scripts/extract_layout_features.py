import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from collections import defaultdict
from math import sqrt
from scipy.spatial.distance import pdist, squareform
# import networkx as nx  # 暂时注释掉
# from sklearn.cluster import DBSCAN  # 暂时注释掉

def extract_layout_features(cs_xml_path, net_xml_path):
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
            
        # 新增特征：聚类簇数（简化版本，基于距离阈值）
        cluster_count = calculate_simple_cluster_count(coords)
            
    else:
        avg_nn = std_nn = min_nn = 0.0
        max_pairwise_distance = 0.0
        cs_density_std = 0.0
        cluster_count = 0

    # 新增特征：空间覆盖与均衡度指标（简化版本）
    coverage_ratio, max_gap_distance, gini_coefficient = calculate_simple_coverage_features(
        cs_coords, lane_coords
    )
    
    # 新增特征：网络位置指标（简化版本）
    avg_betweenness_centrality = 0.0  # 暂时设为0，等依赖包安装后再实现

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

def calculate_simple_cluster_count(coords):
    """简化版本的聚类计数，基于距离阈值"""
    try:
        if len(coords) <= 1:
            return 0
        
        # 使用简单的距离阈值方法
        threshold = 1000  # 1000米阈值
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

# 以下函数暂时注释掉，等依赖包安装后再启用
# def calculate_network_centrality(cs_coords, lane_coords, net_xml_path):
#     """计算网络中心性指标"""
#     pass

# def build_road_network(net_xml_path):
#     """从net.xml构建路网图"""
#     pass

def extract_all_layout_features(cs_dir, net_xml_path, output_dir=None):
    """批量提取所有充电桩布局的特征"""
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

if __name__ == "__main__":
    # 测试单个文件
    sample = extract_layout_features(
        cs_xml_path="/home/ubuntu/project/MSC/Msc_Project/sumo/dataset_1/S001/cs/cs_group_001.xml", # S001需要修改为实际的scenario_id
        net_xml_path="/home/ubuntu/project/MSC/Msc_Project/data/map/glasgow_clean.net.xml"
    )

    print("Extracted layout features:")
    for k, v in sample.items():
        print(f"{k}: {v}")

    # 或写入 CSV
    df = pd.DataFrame([sample])
    df.to_csv("sumo/dataset_1/S001/cs/layout_features_sample.csv", index=False)
    
    # 批量处理所有文件
    # extract_all_layout_features(
    #     cs_dir="/home/ubuntu/project/MSC/Msc_Project/data/cs",
    #     net_xml_path="/home/ubuntu/project/MSC/Msc_Project/data/map/glasgow_clean.net.xml"
    # )


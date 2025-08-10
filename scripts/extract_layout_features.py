import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from collections import defaultdict
from math import sqrt
from scipy.spatial.distance import pdist, squareform

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
            
    else:
        avg_nn = std_nn = min_nn = 0.0
        max_pairwise_distance = 0.0
        cs_density_std = 0.0

    # 构建输出字典
    features = {
        "cs_count": cs_count,
        "avg_dist_to_center": avg_dist_to_center,
        "avg_nearest_neighbor": avg_nn,
        "std_nearest_neighbor": std_nn,
        "min_distance": min_nn,
        "max_pairwise_distance": max_pairwise_distance,
        "cs_density_std": cs_density_std
    }

    return features

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


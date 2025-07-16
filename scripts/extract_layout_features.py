import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from collections import defaultdict
from math import sqrt
from scipy.spatial.distance import pdist

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
    else:
        avg_nn = std_nn = min_nn = 0.0

    # 构建输出字典
    features = {
        "cs_count": cs_count,
        "avg_dist_to_center": avg_dist_to_center,
        "avg_nearest_neighbor": avg_nn,
        "std_nearest_neighbor": std_nn,
        "min_distance": min_nn
    }

    return features

if __name__ == "__main__":
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


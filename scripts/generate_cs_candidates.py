import os
import json
import random
import xml.etree.ElementTree as ET

def generate_cs_candidates(net_file, output_file, layout_count, cs_per_layout):
    valid_type_whitelist = {
        "highway.living_street", "highway.primary", "highway.primary_link",
        "highway.residential", "highway.secondary", "highway.secondary_link",
        "highway.tertiary", "highway.tertiary_link", "highway.trunk",
        "highway.unclassified"
    }
    tree = ET.parse(net_file)
    root = tree.getroot()
    roundabout_edges = set()
    deadend_junctions = set()
    for elem in root.findall("roundabout"):
        edges = elem.get("edges", "")
        roundabout_edges.update(edges.split())
    for junction in root.findall("junction"):
        if junction.get("type") == "dead_end":
            deadend_junctions.add(junction.get("id"))
    edge_to_junction = {}
    for edge in root.findall("edge"):
        if edge.get("function") == "internal" or edge.get("id", "").startswith(":"):
            continue
        to = edge.get("to")
        if to:
            edge_to_junction[edge.get("id")] = to
    # 收集合法edge及其lane长度
    valid_edges = []
    edge_lane_length = {}  # edge_id -> (lane_id, length)
    for edge in root.findall("edge"):
        edge_id = edge.get("id", "")
        edge_func = edge.get("function", "")
        edge_type = edge.get("type", "")
        if edge_func == "internal" or edge_id.startswith(":"):
            continue
        if edge_id in roundabout_edges:
            continue
        if edge_to_junction.get(edge_id) in deadend_junctions:
            continue
        if edge_type not in valid_type_whitelist:
            continue
        for lane in edge.findall("lane"):
            length = float(lane.get("length", "0"))
            allow = lane.get("allow", "")
            # 只保留长度足够的lane
            if length >= 10 and ("passenger" in allow or allow == ""):
                valid_edges.append(edge_id)
                edge_lane_length[edge_id] = (lane.get("id"), length)
                break
    valid_edges = list(set(valid_edges))
    if len(valid_edges) < cs_per_layout:
        raise ValueError("候选 edge 数量不足，无法生成布局。")
    layout_registry = {}
    for i in range(1, layout_count + 1):
        selected = random.sample(valid_edges, cs_per_layout)
        layout_key = f"cs_group_{i:03d}"
        layout_registry[layout_key] = []
        for edge_id in selected:
            lane_id, lane_length = edge_lane_length[edge_id]
            # 充电桩长度5米，pos范围[0, lane_length-5]，如lane_length<=5则跳过
            max_pos = max(lane_length - 5.0, 0.0)
            if max_pos <= 0.0:
                continue
            pos = round(random.uniform(0.0, max_pos), 1)
            layout_registry[layout_key].append({
                "edge_id": edge_id,
                "pos": pos
            })
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(layout_registry, f, indent=2)
    print(f"✅ Generated {layout_count} layouts, saved to {output_file}")

if __name__ == "__main__":
    net_file = "/home/ubuntu/project/MSC/Msc_Project/data/map/glasgow_clean.net.xml"
    output_file = "/home/ubuntu/project/MSC/Msc_Project/data/dataset_1/layout_registry.json"
    layout_count = 10
    cs_per_layout = 215
    generate_cs_candidates(net_file, output_file, layout_count, cs_per_layout)

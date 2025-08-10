import os
import sys
import json
import argparse
import xml.etree.ElementTree as ET
from generate_cs_candidates import generate_cs_candidates
from extract_layout_features import extract_layout_features, extract_all_layout_features

def generate_global_cs_layouts(n_layouts, cs_count, net_file, output_dir):
    """
    生成n个充电站布局作为全局资源
    
    Args:
        n_layouts: 要生成的布局数量
        cs_count: 每个布局的充电站数量
        net_file: 网络文件路径
        output_dir: 输出目录
    """
    print(f"🚀 开始生成 {n_layouts} 个充电站布局")
    print(f"📊 每个布局包含 {cs_count} 个充电站")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查是否已有XML文件存在
    existing_xml_files = []
    for i in range(n_layouts):
        layout_id = f"cs_group_{i+1:03d}"
        cs_xml_file = os.path.join(output_dir, f"{layout_id}.xml")
        if os.path.exists(cs_xml_file):
            existing_xml_files.append(layout_id)
    
    if existing_xml_files:
        print(f"📋 发现已存在的XML文件: {len(existing_xml_files)} 个")
        print("🔄 将跳过XML生成，只更新layout features")
        
        # 直接更新所有layout features
        print("📊 开始更新layout features...")
        extract_all_layout_features(output_dir, net_file, output_dir)
        
        # 读取现有的注册表
        registry_file = os.path.join(output_dir, "layout_registry.json")
        if os.path.exists(registry_file):
            with open(registry_file, 'r') as f:
                layout_registry = json.load(f)
        else:
            layout_registry = {}
        
        print(f"✅ Layout features更新完成")
        return layout_registry
    
    # 如果没有现有文件，则生成新的布局
    print("📍 生成候选充电站位置...")
    candidates_file = os.path.join(output_dir, "cs_candidates.json")
    generate_cs_candidates(net_file, candidates_file, n_layouts, cs_count)
    
    # 读取候选位置
    with open(candidates_file, 'r') as f:
        candidates = json.load(f)
    
    # 生成n个不同的布局
    layout_registry = {}
    
    for i in range(n_layouts):
        layout_id = f"cs_group_{i+1:03d}"
        print(f"🎯 生成布局 {layout_id} ({i+1}/{n_layouts})")
        
        # 生成充电站XML文件
        cs_xml_file = os.path.join(output_dir, f"{layout_id}.xml")
        
        # 创建临时场景配置
        temp_config = {
            'scenario_id': layout_id,
            'cs_layout_id': layout_id,
            'cs_count': cs_count
        }
        
        # 直接生成XML文件
        root = ET.Element("additional")
        
        # 从candidates中获取当前布局的充电站位置
        layout_candidates = candidates[layout_id]
        
        for i, site in enumerate(layout_candidates):
            edge_id = site["edge_id"]
            pos = float(site["pos"])
            station_id = f"cs_{i+1:03d}"
            
            ET.SubElement(root, "chargingStation", attrib={
                "id": station_id,
                "lane": f"{edge_id}_0",
                "startPos": str(pos),
                "endPos": str(pos + 5.0),
                "power": "22000",  # 使用默认值
                "efficiency": "0.95",
                "chargeDelay": "1.0",
                "chargeInTransit": "0"
            })
        
        # 保存XML文件
        tree = ET.ElementTree(root)
        tree.write(cs_xml_file, encoding="utf-8", xml_declaration=True)
        
        # 提取布局特征
        features_file = os.path.join(output_dir, f"{layout_id}_layout_features.csv")
        features = extract_layout_features(cs_xml_file, net_file)
        
        # 保存特征到CSV
        import pandas as pd
        df = pd.DataFrame([features])
        df.to_csv(features_file, index=False)
        
        # 记录到注册表
        layout_registry[layout_id] = {
            "cs_xml_file": f"{layout_id}.xml",
            "features_file": f"{layout_id}_layout_features.csv",
            "cs_count": cs_count
        }
        
        print(f"✅ 布局 {layout_id} 生成完成")
    
    # 保存布局注册表
    registry_file = os.path.join(output_dir, "layout_registry.json")
    with open(registry_file, 'w') as f:
        json.dump(layout_registry, f, indent=2)
    
    print(f"\n🎉 充电站布局生成完成！")
    print(f"📁 输出目录: {output_dir}")
    print(f"📋 布局注册表: {registry_file}")
    print(f"📊 生成了 {n_layouts} 个布局:")
    for layout_id in layout_registry.keys():
        print(f"   - {layout_id}")
    
    return layout_registry

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成全局充电站布局')
    parser.add_argument('-n', '--n_layouts', type=int, default=10, 
                       help='要生成的布局数量 (默认: 10)')
    parser.add_argument('-c', '--cs_count', type=int, default=215, 
                       help='每个布局的充电站数量 (默认: 215)')
    parser.add_argument('--net_file', type=str, 
                       default='data/map/glasgow_clean.net.xml',
                       help='网络文件路径')
    parser.add_argument('--output_dir', type=str, 
                       default='data/cs',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 生成布局
    layout_registry = generate_global_cs_layouts(
        args.n_layouts, 
        args.cs_count, 
        args.net_file, 
        args.output_dir
    ) 
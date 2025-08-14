import os
import json
import argparse
import xml.etree.ElementTree as ET
from load_scenario import load_scenario


def generate_charging_stations(scenario_id: str,
                                layout_path: str = "data/dataset_1/layout_registry.json",
                                scenario_path: str = "data/dataset_1/scenario_matrix.csv",
                                out_dir_base: str = "/home/ubuntu/project/MSC/Msc_Project/sumo/dataset_1"):
    # Load scenario config
    config = load_scenario(scenario_id, scenario_path)
    cs_layout_id = config["cs_layout_id"]  # e.g., cs_group_001
    cs_count = config["cs_count"]

    # Load layout candidates
    with open(layout_path, "r") as f:
        layout_registry = json.load(f)

    if cs_layout_id not in layout_registry:
        raise ValueError(f"CS layout id '{cs_layout_id}' not found in layout registry")

    layout = layout_registry[cs_layout_id]
    if len(layout) < cs_count:
        raise ValueError(f"Layout '{cs_layout_id}' only has {len(layout)} locations, fewer than required {cs_count}")

    selected_sites = layout[:cs_count]

    # Generate XML
    root = ET.Element("additional")

    for i, site in enumerate(selected_sites):
        edge_id = site["edge_id"]
        pos = float(site["pos"])
        station_id = f"cs_{i+1:03d}"

        ET.SubElement(root, "chargingStation", attrib={
            "id": station_id,
            "lane": f"{edge_id}_0",
            "startPos": str(pos),
            "endPos": str(pos + 5.0),
            "power": "220000.0",
            "efficiency": "0.95",
            "chargeDelay": "1.0",
            "chargeInTransit": "0"
            # "chargingVehicles": ""
        })

    # Write output
    out_dir = os.path.join(out_dir_base, scenario_id, "cs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{cs_layout_id}.xml")

    tree = ET.ElementTree(root)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"✅ Charging stations for scenario '{scenario_id}' saved to: {out_path}")


def generate_charging_stations_from_json(json_file_path, output_dir=None):
    """从JSON候选文件批量生成XML充电桩文件"""
    if output_dir is None:
        output_dir = os.path.dirname(json_file_path)
    
    print(f"🚀 开始从JSON文件生成充电桩XML文件")
    print(f"📊 JSON文件: {json_file_path}")
    print(f"💾 输出目录: {output_dir}")
    
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            cs_data = json.load(f)
        
        print(f"📊 加载了 {len(cs_data)} 个充电桩布局组")
        
        success_count = 0
        total_count = len(cs_data)
        
        for idx, (group_id, group_data) in enumerate(cs_data.items(), 1):
            print(f"\n[{idx}/{total_count}] 处理布局组: {group_id}")
            
            try:
                # 生成XML
                root = ET.Element("additional")
                
                for i, site in enumerate(group_data):
                    edge_id = site["edge_id"]
                    pos = float(site["pos"])
                    station_id = f"cs_{i+1:03d}"
                    
                    # 添加充电桩元素，使用与参考文件相同的格式
                    ET.SubElement(root, "chargingStation", attrib={
                        "id": station_id,
                        "lane": f"{edge_id}_0",  # 添加_0后缀表示第0条车道
                        "startPos": str(pos),
                        "endPos": str(pos + 5.0),  # endPos = startPos + 5.0
                        "power": "22000",  # 22kW功率
                        "efficiency": "0.95",  # 95%效率
                        "chargeDelay": "1.0",
                        "chargeInTransit": "0"
                    })
                
                # 创建输出目录
                os.makedirs(output_dir, exist_ok=True)
                
                # 写入XML文件
                out_path = os.path.join(output_dir, f"{group_id}.xml")
                tree = ET.ElementTree(root)
                
                # 使用与参考文件相同的格式写入
                tree.write(out_path, encoding="utf-8", xml_declaration=True)
                
                success_count += 1
                print(f"✅ 生成XML完成: {group_id} ({len(group_data)} 个充电桩) -> {out_path}")
                
            except Exception as e:
                print(f"❌ 生成XML失败 {group_id}: {e}")
                continue
        
        print(f"\n🎉 批量XML生成完成！")
        print(f"✅ 成功生成: {success_count}/{total_count} 个XML文件")
        print(f"💾 文件保存到: {output_dir}")
        
        return success_count
        
    except Exception as e:
        print(f"❌ 处理JSON文件失败: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description='生成充电桩XML文件')
    parser.add_argument('--json_file', type=str,
                       help='JSON布局文件路径 (如 cs_candidates_51-100.json)')
    parser.add_argument('--output_dir', type=str,
                       help='输出目录路径 (默认: JSON文件所在目录)')
    parser.add_argument('--scenario_id', type=str,
                       help='单个场景ID (传统模式，如 S001)')
    parser.add_argument('--layout_path', type=str,
                       default="data/dataset_1/layout_registry.json",
                       help='布局注册表文件路径 (传统模式)')
    parser.add_argument('--scenario_path', type=str,
                       default="data/dataset_1/scenario_matrix.csv",
                       help='场景矩阵文件路径 (传统模式)')
    parser.add_argument('--out_dir_base', type=str,
                       default="/home/ubuntu/project/MSC/Msc_Project/sumo/dataset_1",
                       help='输出基础目录 (传统模式)')
    
    args = parser.parse_args()
    
    if args.json_file:
        # JSON文件批量生成模式
        if not os.path.exists(args.json_file):
            print(f"❌ JSON文件不存在: {args.json_file}")
            parser.print_help()
            print("\n💡 使用示例:")
            print("   # 从JSON文件批量生成XML")
            print("   python scripts/generator_charging_site.py --json_file data/cs_51-100/cs_candidates_51-100.json")
            print("\n   # 指定输出目录")
            print("   python scripts/generator_charging_site.py --json_file data/cs_51-100/cs_candidates_51-100.json --output_dir data/cs_51-100")
            print("\n   # 传统单个场景模式")
            print("   python scripts/generator_charging_site.py --scenario_id S001")
            exit(1)
        
        output_dir = args.output_dir if args.output_dir else os.path.dirname(args.json_file)
        generate_charging_stations_from_json(args.json_file, output_dir)
        
    elif args.scenario_id:
        # 传统单个场景生成模式
        generate_charging_stations(
            scenario_id=args.scenario_id,
            layout_path=args.layout_path,
            scenario_path=args.scenario_path,
            out_dir_base=args.out_dir_base
        )
    else:
        # 默认显示帮助
        parser.print_help()
        print("\n💡 使用示例:")
        print("   # 从JSON文件批量生成XML")
        print("   python scripts/generator_charging_site.py --json_file data/cs_51-100/cs_candidates_51-100.json")
        print("\n   # 传统单个场景模式")
        print("   python scripts/generator_charging_site.py --scenario_id S001")


if __name__ == "__main__":
    main()

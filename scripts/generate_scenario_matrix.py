import os
import csv
import argparse
import json

def generate_scenario_matrix(layout_registry_file, output_file, rou_types=['sequence', 'mixed', 'random']):
    """
    生成简化的场景矩阵文件
    
    Args:
        layout_registry_file: 布局注册表文件路径
        output_file: 输出的场景矩阵文件路径
        rou_types: 路由类型列表
    """
    print(f"🚀 生成场景矩阵文件")
    
    # 读取布局注册表
    with open(layout_registry_file, 'r') as f:
        layout_registry = json.load(f)
    
    # 生成场景矩阵
    scenarios = []
    scenario_id = 1
    
    for layout_id in layout_registry.keys():
        for rou_type in rou_types:
            scenario = {
                'scenario_id': f'S{scenario_id:03d}',
                'cs_layout_id': layout_id,
                'rou_type': rou_type
            }
            scenarios.append(scenario)
            scenario_id += 1
    
    # 写入CSV文件
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['scenario_id', 'cs_layout_id', 'rou_type'])
        writer.writeheader()
        writer.writerows(scenarios)
    
    print(f"✅ 场景矩阵生成完成: {output_file}")
    print(f"📊 生成了 {len(scenarios)} 个场景:")
    print(f"   - 布局数量: {len(layout_registry)}")
    print(f"   - 路由类型: {len(rou_types)}")
    print(f"   - 总场景数: {len(layout_registry) * len(rou_types)}")
    
    return scenarios

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成场景矩阵文件')
    parser.add_argument('--layout_registry', type=str, 
                       default='data/cs/layout_registry.json',
                       help='布局注册表文件路径')
    parser.add_argument('--output', type=str, 
                       default='data/scenario_matrix.csv',
                       help='输出的场景矩阵文件路径')
    parser.add_argument('--rou_types', nargs='+', 
                       default=['sequence', 'mixed', 'random'],
                       help='路由类型列表')
    
    args = parser.parse_args()
    
    # 生成场景矩阵
    scenarios = generate_scenario_matrix(
        args.layout_registry, 
        args.output, 
        args.rou_types
    ) 
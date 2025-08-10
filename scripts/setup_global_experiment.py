#!/usr/bin/env python3
"""
全局实验设置脚本
一次性生成所有需要的全局资源
"""

import os
import sys
import argparse
from generate_three_route_types import create_three_route_types
from generator_trip import generate_trip_routes, load_scenario
from generate_global_cs_layouts import generate_global_cs_layouts
from generate_scenario_matrix import generate_scenario_matrix
from vehicle_config import get_battery_config

def setup_global_experiment(vehicle_count, ev_ratio, n_layouts, cs_count):
    """
    设置全局实验环境
    
    Args:
        vehicle_count: 车辆总数
        ev_ratio: EV占比
        n_layouts: 充电站布局数量
        cs_count: 每个布局的充电站数量
    """
    print("🚀 开始设置全局实验环境")
    print("=" * 50)
    
    # 定义文件路径
    sumo_net_file = 'data/map/glasgow_clean.net.xml'
    routes_dir = 'data/routes'
    cs_dir = 'data/cs'
    
    # 1. 生成全局路由文件
    print("\n📁 步骤1: 生成全局路由文件")
    print("-" * 30)
    
    rou_files = {}
    
    # 检查三种路由文件是否已存在
    route_files_exist = True
    for mode in ['sequence', 'mixed', 'random']:
        route_file = os.path.join(routes_dir, f"{mode}.rou.xml")
        if not os.path.exists(route_file):
            route_files_exist = False
            break
        rou_files[mode] = route_file
    
    if route_files_exist:
        print("✅ 三种路由文件已存在，跳过生成")
    else:
        print("🔄 生成三种路由文件...")
        
        # 使用generator_trip.py生成基础rou.xml
        vehicle_config = get_battery_config()
        
        # 创建临时场景配置
        temp_config = {
            'scenario_id': 'TEMP',
            'vehicle_count': vehicle_count,
            'ev_ratio': ev_ratio
        }
        
        temp_routes_dir = 'temp_routes'
        vehicles_add_file = 'data/vehicles.add.xml'
        
        sumo_home = os.environ.get('SUMO_HOME', '/usr/share/sumo')
        tools = os.path.join(sumo_home, 'tools')
        duarouter_path = 'duarouter'
        
        # 生成基础rou.xml
        base_rou_file = generate_trip_routes(
            temp_config, sumo_net_file, temp_routes_dir, tools, 
            vehicles_add_file, duarouter_path, vehicle_config=vehicle_config
        )
        
        # 基于基础rou.xml生成三种模式
        rou_files = create_three_route_types(base_rou_file, routes_dir)
        
        # 清理临时文件
        import shutil
        if os.path.exists(temp_routes_dir):
            shutil.rmtree(temp_routes_dir)
    
    # 2. 生成充电站布局
    print("\n📁 步骤2: 生成充电站布局")
    print("-" * 30)
    
    cs_dir = 'data/cs'
    layout_registry_file = os.path.join(cs_dir, "layout_registry.json")
    
    # 检查布局注册表是否已存在
    if os.path.exists(layout_registry_file):
        print("✅ 充电站布局已存在，跳过生成")
        with open(layout_registry_file, 'r') as f:
            import json
            layout_registry = json.load(f)
    else:
        print("🔄 生成充电站布局...")
        layout_registry = generate_global_cs_layouts(
            n_layouts, cs_count, sumo_net_file, cs_dir
        )
    
    # 3. 生成场景矩阵
    print("\n📁 步骤3: 生成场景矩阵")
    print("-" * 30)
    
    scenario_matrix_file = 'data/scenario_matrix.csv'
    
    # 检查场景矩阵是否已存在
    if os.path.exists(scenario_matrix_file):
        print("✅ 场景矩阵已存在，跳过生成")
        scenarios = []  # 这里可以读取现有的场景矩阵
    else:
        print("🔄 生成场景矩阵...")
        scenarios = generate_scenario_matrix(
            layout_registry_file, scenario_matrix_file
        )
    
    # 4. 总结
    print("\n🎉 全局实验环境设置完成！")
    print("=" * 50)
    print(f"📊 实验配置:")
    print(f"   - 车辆总数: {vehicle_count}")
    print(f"   - EV占比: {ev_ratio}")
    print(f"   - 充电站布局数: {n_layouts}")
    print(f"   - 每个布局充电站数: {cs_count}")
    print(f"   - 总场景数: {len(scenarios)}")
    
    print(f"\n📁 生成的文件:")
    print(f"   - 路由文件: {routes_dir}/")
    print(f"     * sequence.rou.xml")
    print(f"     * mixed.rou.xml")
    print(f"     * random.rou.xml")
    print(f"   - 充电站布局: {cs_dir}/")
    print(f"     * cs_group_001.xml ~ cs_group_{n_layouts:03d}.xml")
    print(f"     * layout_registry.json")
    print(f"   - 场景矩阵: {scenario_matrix_file}")
    
    print(f"\n🚀 下一步操作:")
    print(f"   # 运行单个场景")
    print(f"   python scripts/run_global_simulation.py -s S001")
    print(f"   # 运行所有场景")
    print(f"   python scripts/run_global_simulation.py")
    
    return {
        'rou_files': rou_files,
        'layout_registry': layout_registry,
        'scenarios': scenarios
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='设置全局实验环境')
    parser.add_argument('--vehicle_count', type=int, default=10000,
                       help='车辆总数 (默认: 10000)')
    parser.add_argument('--ev_ratio', type=float, default=0.2,
                       help='EV占比 (默认: 0.2)')
    parser.add_argument('--n_layouts', type=int, default=10,
                       help='充电站布局数量 (默认: 10)')
    parser.add_argument('--cs_count', type=int, default=215,
                       help='每个布局的充电站数量 (默认: 215)')
    
    args = parser.parse_args()
    
    # 设置实验环境
    result = setup_global_experiment(
        args.vehicle_count,
        args.ev_ratio,
        args.n_layouts,
        args.cs_count
    ) 
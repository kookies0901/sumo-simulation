import os
import sys
import random
import xml.etree.ElementTree as ET
from generator_trip import generate_trip_routes, load_scenario
from vehicle_config import get_battery_config

def create_three_route_types(base_rou_file, output_dir):
    """
    基于一个基础的rou.xml文件，生成三种不同出发模式的rou.xml文件
    
    Args:
        base_rou_file: 基础rou.xml文件路径
        output_dir: 输出目录
    """
    print(f"🚀 基于 {base_rou_file} 生成三种出发模式")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取基础rou.xml文件
    tree = ET.parse(base_rou_file)
    root = tree.getroot()
    
    # 收集所有车辆
    vehicles = []
    for vehicle in root.findall('vehicle'):
        vehicles.append(vehicle)
    
    print(f"📊 找到 {len(vehicles)} 辆车")
    
    # 分离EV和Petrol车辆
    ev_vehicles = []
    petrol_vehicles = []
    
    for vehicle in vehicles:
        vid = vehicle.get('id', '')
        if vid.startswith('EV_'):
            ev_vehicles.append(vehicle)
        elif vid.startswith('PET_'):
            petrol_vehicles.append(vehicle)
    
    print(f"📊 EV车辆: {len(ev_vehicles)}, Petrol车辆: {len(petrol_vehicles)}")
    
    # 生成三种模式
    modes = ['sequence', 'mixed', 'random']
    
    for mode in modes:
        print(f"\n🎯 生成 {mode} 模式...")
        
        # 创建新的routes元素
        new_root = ET.Element('routes')
        
        if mode == 'sequence':
            # 模式1：先EV后Petrol，依次出发
            current_time = 0
            for vehicle in ev_vehicles:
                # 复制车辆元素
                new_vehicle = ET.SubElement(new_root, 'vehicle')
                for key, value in vehicle.attrib.items():
                    if key == 'depart':
                        new_vehicle.set('depart', str(current_time))
                    else:
                        new_vehicle.set(key, value)
                
                # 复制子元素（route和param）
                for child in vehicle:
                    new_vehicle.append(child)
                
                current_time += 1
            
            # 然后添加Petrol车辆
            for vehicle in petrol_vehicles:
                new_vehicle = ET.SubElement(new_root, 'vehicle')
                for key, value in vehicle.attrib.items():
                    if key == 'depart':
                        new_vehicle.set('depart', str(current_time))
                    else:
                        new_vehicle.set(key, value)
                
                for child in vehicle:
                    new_vehicle.append(child)
                
                current_time += 1
                
        elif mode == 'mixed':
            # 模式2：混合出发，每秒依次出发
            all_vehicles = ev_vehicles + petrol_vehicles
            random.seed(42)  # 固定随机种子
            random.shuffle(all_vehicles)  # 随机打乱顺序
            
            for i, vehicle in enumerate(all_vehicles):
                new_vehicle = ET.SubElement(new_root, 'vehicle')
                for key, value in vehicle.attrib.items():
                    if key == 'depart':
                        new_vehicle.set('depart', str(i))  # 第i秒出发
                    else:
                        new_vehicle.set(key, value)
                
                for child in vehicle:
                    new_vehicle.append(child)
                    
        elif mode == 'random':
            # 模式3：随机混合出发
            all_vehicles = ev_vehicles + petrol_vehicles
            random.seed(42)  # 固定随机种子
            random.shuffle(all_vehicles)  # 随机打乱顺序
            
            # 随机分配出发时间
            total_vehicles = len(all_vehicles)
            departure_times = list(range(total_vehicles))
            random.shuffle(departure_times)
            
            for i, vehicle in enumerate(all_vehicles):
                new_vehicle = ET.SubElement(new_root, 'vehicle')
                for key, value in vehicle.attrib.items():
                    if key == 'depart':
                        new_vehicle.set('depart', str(departure_times[i]))  # 随机出发时间
                    else:
                        new_vehicle.set(key, value)
                
                for child in vehicle:
                    new_vehicle.append(child)
        
        # 保存新的rou.xml文件
        output_file = os.path.join(output_dir, f"{mode}.rou.xml")
        ET.ElementTree(new_root).write(output_file, encoding='utf-8', xml_declaration=True)
        print(f"✅ 生成 {mode} 模式: {output_file}")
    
    print(f"\n🎉 三种模式生成完成！")
    return [os.path.join(output_dir, f"{mode}.rou.xml") for mode in modes]

def main():
    """主函数"""
    print("🚀 生成三种出发模式的路由文件")
    
    # 使用generator_trip.py生成基础rou.xml
    print("\n📁 步骤1: 生成基础路由文件")
    print("-" * 30)
    
    # 加载车辆配置
    vehicle_config = get_battery_config()
    
    # 加载场景配置
    cfg = load_scenario('S001', 'data/dataset_1/scenario_matrix.csv')
    net = os.path.join('data', 'map', 'glasgow_clean.net.xml')
    out = os.path.join('temp_routes')  # 临时目录
    vehicles_add = os.path.join('data', 'vehicles.add.xml')
    
    # SUMO工具路径
    sumo_home = os.environ.get('SUMO_HOME', '/usr/share/sumo')
    tools = os.path.join(sumo_home, 'tools')
    duarouter_path = 'duarouter'
    
    # 生成基础rou.xml
    base_rou_file = generate_trip_routes(cfg, net, out, tools, vehicles_add, duarouter_path, vehicle_config=vehicle_config)
    print(f"✅ 基础路由文件生成完成: {base_rou_file}")
    
    # 生成三种模式
    print("\n📁 步骤2: 生成三种出发模式")
    print("-" * 30)
    
    output_dir = 'data/routes'
    rou_files = create_three_route_types(base_rou_file, output_dir)
    
    # 清理临时文件
    import shutil
    if os.path.exists(out):
        shutil.rmtree(out)
    
    print(f"\n🎉 所有路由文件生成完成！")
    print(f"📁 输出目录: {output_dir}")
    for rou_file in rou_files:
        print(f"   - {os.path.basename(rou_file)}")

if __name__ == '__main__':
    main() 
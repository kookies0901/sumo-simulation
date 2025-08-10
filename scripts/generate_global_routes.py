import os
import subprocess
import sys
import random
import xml.etree.ElementTree as ET
from vehicle_config import get_battery_config

def remove_vtype_from_rou(rou_path):
    tree = ET.parse(rou_path)
    root = tree.getroot()
    vtypes = root.findall('vType')
    for vtype in vtypes:
        root.remove(vtype)
    tree.write(rou_path, encoding='utf-8', xml_declaration=True)

def add_random_ev_charge_level(rou_path: str, vehicle_config=None):
    """
    为EV车辆添加随机的初始电量参数
    """
    if vehicle_config is None:
        vehicle_config = get_battery_config()
    
    capacity_wh = vehicle_config["capacity"]
    min_soc = vehicle_config["needToChargeLevel"]
    max_soc = vehicle_config["saturatedChargeLevel"]
    
    tree = ET.parse(rou_path)
    root = tree.getroot()
    
    for vehicle in root.findall('vehicle'):
        vid = vehicle.get('id', '')
        if vid.startswith('EV_'):
            # 调整为needToChargeLevel和0.8之间的随机值
            random_soc = random.uniform(min_soc, 1)
            charge_level = int(random_soc * capacity_wh)
            
            param = ET.SubElement(vehicle, 'param')
            param.set('key', 'device.battery.chargeLevel')
            param.set('value', str(charge_level))
    
    tree.write(rou_path, encoding='utf-8', xml_declaration=True)

def generate_sequence_departures(ev_count, pet_count, sumo_net_file, sumo_tools_path, output_dir, fringe_factor):
    """模式1：先EV后Petrol，依次出发"""
    print(f"🎯 生成顺序出发模式: {ev_count}辆EV + {pet_count}辆Petrol")
    
    # 生成EV trips
    ev_trips = os.path.join(output_dir, "ev.trips.xml")
    cmd_ev = [sys.executable, os.path.join(sumo_tools_path, 'randomTrips.py'),
              '-n', sumo_net_file,
              '--fringe-factor', str(fringe_factor),
              '--validate',  # 验证OD对的有效性
              '-o', ev_trips,
              '-e', str(ev_count),
              '--prefix', 'EV_',
              '-b', '0',
              '-p', str(ev_count)]
    subprocess.run(cmd_ev, check=True)
    
    # 生成Petrol trips
    pet_trips = os.path.join(output_dir, "pet.trips.xml")
    cmd_pet = [sys.executable, os.path.join(sumo_tools_path, 'randomTrips.py'),
               '-n', sumo_net_file,
               '--fringe-factor', str(fringe_factor),
               '--validate',  # 验证OD对的有效性
               '-o', pet_trips,
               '-e', str(pet_count),
               '--prefix', 'PET_',
               '-b', str(ev_count),
               '-p', str(pet_count)]
    subprocess.run(cmd_pet, check=True)
    
    # 合并trips文件
    merged_trips = os.path.join(output_dir, "sequence.trips.xml")
    merge_trips_xml([ev_trips, pet_trips], merged_trips)
    
    return merged_trips

def generate_mixed_departures(ev_count, pet_count, sumo_net_file, sumo_tools_path, output_dir, fringe_factor):
    """模式2：混合出发，每秒依次出发"""
    print(f"🎯 生成混合出发模式: {ev_count}辆EV + {pet_count}辆Petrol")
    
    # 先生成固定的OD对
    ev_trips = os.path.join(output_dir, "ev.trips.xml")
    cmd_ev = [sys.executable, os.path.join(sumo_tools_path, 'randomTrips.py'),
              '-n', sumo_net_file,
              '--fringe-factor', str(fringe_factor),
              '--validate',  # 验证OD对的有效性
              '-o', ev_trips,
              '-e', str(ev_count),
              '--prefix', 'EV_',
              '-b', '0',
              '-p', str(ev_count)]
    subprocess.run(cmd_ev, check=True)
    
    pet_trips = os.path.join(output_dir, "pet.trips.xml")
    cmd_pet = [sys.executable, os.path.join(sumo_tools_path, 'randomTrips.py'),
               '-n', sumo_net_file,
               '--fringe-factor', str(fringe_factor),
               '--validate',  # 验证OD对的有效性
               '-o', pet_trips,
               '-e', str(pet_count),
               '--prefix', 'PET_',
               '-b', str(ev_count),
               '-p', str(pet_count)]
    subprocess.run(cmd_pet, check=True)
    
    # 读取所有OD对
    ev_tree = ET.parse(ev_trips)
    pet_tree = ET.parse(pet_trips)
    ev_root = ev_tree.getroot()
    pet_root = pet_tree.getroot()
    
    all_od_pairs = []
    for trip in ev_root.findall('trip'):
        all_od_pairs.append({
            'from': trip.get('from'),
            'to': trip.get('to')
        })
    for trip in pet_root.findall('trip'):
        all_od_pairs.append({
            'from': trip.get('from'),
            'to': trip.get('to')
        })
    
    print(f"📊 收集到 {len(all_od_pairs)} 个OD对 (需要 {ev_count + pet_count} 个)")
    
    # 检查OD对数量是否足够
    if len(all_od_pairs) < ev_count + pet_count:
        print(f"⚠️ 警告: OD对数量不足，尝试重新生成...")
        # 重新生成，不使用--validate参数
        cmd_ev_no_validate = [sys.executable, os.path.join(sumo_tools_path, 'randomTrips.py'),
                             '-n', sumo_net_file,
                             '--fringe-factor', str(fringe_factor),
                             '-o', ev_trips,
                             '-e', str(ev_count),
                             '--prefix', 'EV_',
                             '-b', '0',
                             '-p', str(ev_count)]
        subprocess.run(cmd_ev_no_validate, check=True)
        
        cmd_pet_no_validate = [sys.executable, os.path.join(sumo_tools_path, 'randomTrips.py'),
                              '-n', sumo_net_file,
                              '--fringe-factor', str(fringe_factor),
                              '-o', pet_trips,
                              '-e', str(pet_count),
                              '--prefix', 'PET_',
                              '-b', str(ev_count),
                              '-p', str(pet_count)]
        subprocess.run(cmd_pet_no_validate, check=True)
        
        # 重新读取OD对
        ev_tree = ET.parse(ev_trips)
        pet_tree = ET.parse(pet_trips)
        ev_root = ev_tree.getroot()
        pet_root = pet_tree.getroot()
        
        all_od_pairs = []
        for trip in ev_root.findall('trip'):
            all_od_pairs.append({
                'from': trip.get('from'),
                'to': trip.get('to')
            })
        for trip in pet_root.findall('trip'):
            all_od_pairs.append({
                'from': trip.get('from'),
                'to': trip.get('to')
            })
        
        print(f"📊 重新生成后收集到 {len(all_od_pairs)} 个OD对")
    
    # 创建新的routes元素
    routes = ET.Element('routes')
    
    # 随机分配车辆类型和出发时间
    random.seed(42)
    
    ev_ids = [f"EV_{i:06d}" for i in range(ev_count)]
    pet_ids = [f"PET_{i:06d}" for i in range(pet_count)]
    
    all_vehicles = ev_ids + pet_ids
    random.shuffle(all_vehicles)
    
    total_vehicles = ev_count + pet_count
    for i, veh_id in enumerate(all_vehicles):
        if i >= len(all_od_pairs):
            print(f"❌ 错误: OD对数量仍然不足，跳过车辆 {veh_id}")
            continue
            
        new_trip = ET.SubElement(routes, 'trip')
        new_trip.set('id', veh_id)
        new_trip.set('from', all_od_pairs[i]['from'])
        new_trip.set('to', all_od_pairs[i]['to'])
        new_trip.set('depart', str(i))
        
        if veh_id.startswith('EV_'):
            new_trip.set('type', 'EV')
        else:
            new_trip.set('type', 'petrol')
    
    merged_trips = os.path.join(output_dir, "mixed.trips.xml")
    ET.ElementTree(routes).write(merged_trips, xml_declaration=True, encoding='utf-8')
    
    return merged_trips

def generate_random_departures(ev_count, pet_count, sumo_net_file, sumo_tools_path, output_dir, fringe_factor):
    """模式3：随机混合出发"""
    print(f"🎯 生成随机出发模式: {ev_count}辆EV + {pet_count}辆Petrol")
    
    # 先生成固定的OD对
    ev_trips = os.path.join(output_dir, "ev.trips.xml")
    cmd_ev = [sys.executable, os.path.join(sumo_tools_path, 'randomTrips.py'),
              '-n', sumo_net_file,
              '--fringe-factor', str(fringe_factor),
              '--validate',  # 验证OD对的有效性
              '-o', ev_trips,
              '-e', str(ev_count),
              '--prefix', 'EV_',
              '-b', '0',
              '-p', str(ev_count)]
    subprocess.run(cmd_ev, check=True)
    
    pet_trips = os.path.join(output_dir, "pet.trips.xml")
    cmd_pet = [sys.executable, os.path.join(sumo_tools_path, 'randomTrips.py'),
               '-n', sumo_net_file,
               '--fringe-factor', str(fringe_factor),
               '--validate',  # 验证OD对的有效性
               '-o', pet_trips,
               '-e', str(pet_count),
               '--prefix', 'PET_',
               '-b', str(ev_count),
               '-p', str(pet_count)]
    subprocess.run(cmd_pet, check=True)
    
    # 读取所有OD对
    ev_tree = ET.parse(ev_trips)
    pet_tree = ET.parse(pet_trips)
    ev_root = ev_tree.getroot()
    pet_root = pet_tree.getroot()
    
    all_od_pairs = []
    for trip in ev_root.findall('trip'):
        all_od_pairs.append({
            'from': trip.get('from'),
            'to': trip.get('to')
        })
    for trip in pet_root.findall('trip'):
        all_od_pairs.append({
            'from': trip.get('from'),
            'to': trip.get('to')
        })
    
    print(f"📊 收集到 {len(all_od_pairs)} 个OD对 (需要 {ev_count + pet_count} 个)")
    
    # 检查OD对数量是否足够
    if len(all_od_pairs) < ev_count + pet_count:
        print(f"⚠️ 警告: OD对数量不足，尝试重新生成...")
        # 重新生成，不使用--validate参数
        cmd_ev_no_validate = [sys.executable, os.path.join(sumo_tools_path, 'randomTrips.py'),
                             '-n', sumo_net_file,
                             '--fringe-factor', str(fringe_factor),
                             '-o', ev_trips,
                             '-e', str(ev_count),
                             '--prefix', 'EV_',
                             '-b', '0',
                             '-p', str(ev_count)]
        subprocess.run(cmd_ev_no_validate, check=True)
        
        cmd_pet_no_validate = [sys.executable, os.path.join(sumo_tools_path, 'randomTrips.py'),
                              '-n', sumo_net_file,
                              '--fringe-factor', str(fringe_factor),
                              '-o', pet_trips,
                              '-e', str(pet_count),
                              '--prefix', 'PET_',
                              '-b', str(ev_count),
                              '-p', str(pet_count)]
        subprocess.run(cmd_pet_no_validate, check=True)
        
        # 重新读取OD对
        ev_tree = ET.parse(ev_trips)
        pet_tree = ET.parse(pet_trips)
        ev_root = ev_tree.getroot()
        pet_root = pet_tree.getroot()
        
        all_od_pairs = []
        for trip in ev_root.findall('trip'):
            all_od_pairs.append({
                'from': trip.get('from'),
                'to': trip.get('to')
            })
        for trip in pet_root.findall('trip'):
            all_od_pairs.append({
                'from': trip.get('from'),
                'to': trip.get('to')
            })
        
        print(f"📊 重新生成后收集到 {len(all_od_pairs)} 个OD对")
    
    # 创建新的routes元素
    routes = ET.Element('routes')
    
    # 随机分配车辆类型和出发时间
    random.seed(42)
    
    ev_ids = [f"EV_{i:06d}" for i in range(ev_count)]
    pet_ids = [f"PET_{i:06d}" for i in range(pet_count)]
    
    all_vehicles = ev_ids + pet_ids
    random.shuffle(all_vehicles)
    
    total_vehicles = ev_count + pet_count
    departure_times = list(range(total_vehicles))
    random.shuffle(departure_times)
    
    for i, veh_id in enumerate(all_vehicles):
        if i >= len(all_od_pairs):
            print(f"❌ 错误: OD对数量仍然不足，跳过车辆 {veh_id}")
            continue
            
        new_trip = ET.SubElement(routes, 'trip')
        new_trip.set('id', veh_id)
        new_trip.set('from', all_od_pairs[i]['from'])
        new_trip.set('to', all_od_pairs[i]['to'])
        new_trip.set('depart', str(departure_times[i]))
        
        if veh_id.startswith('EV_'):
            new_trip.set('type', 'EV')
        else:
            new_trip.set('type', 'petrol')
    
    merged_trips = os.path.join(output_dir, "random.trips.xml")
    ET.ElementTree(routes).write(merged_trips, xml_declaration=True, encoding='utf-8')
    
    return merged_trips

def merge_trips_xml(input_files: list, output_file: str):
    """合并多个trip XML文件"""
    routes = ET.Element('routes')
    
    for f in input_files:
        tree = ET.parse(f)
        for veh in tree.getroot():
            vid = veh.get('id', '')
            if vid.startswith('EV_'):
                veh.set('type', 'EV')
            elif vid.startswith('PET_'):
                veh.set('type', 'petrol')
            else:
                veh.set('type', 'unknown')
            routes.append(veh)
    
    ET.ElementTree(routes).write(output_file, xml_declaration=True, encoding='utf-8')

def generate_global_routes(vehicle_count, ev_ratio, sumo_net_file, sumo_tools_path, output_dir, vehicles_add_file, duarouter_path):
    """
    生成三种出发模式的全局路由文件
    """
    print(f"🚀 开始生成全局路由文件")
    print(f"📊 车辆总数: {vehicle_count}, EV占比: {ev_ratio}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算车辆数量
    ev_count = int(vehicle_count * ev_ratio)
    pet_count = vehicle_count - ev_count
    
    # 生成三种模式的trips文件
    sequence_trips = generate_sequence_departures(ev_count, pet_count, sumo_net_file, sumo_tools_path, output_dir, 10)
    mixed_trips = generate_mixed_departures(ev_count, pet_count, sumo_net_file, sumo_tools_path, output_dir, 10)
    random_trips = generate_random_departures(ev_count, pet_count, sumo_net_file, sumo_tools_path, output_dir, 10)
    
    # 生成对应的rou.xml文件
    rou_files = {}
    for mode, trips_file in [('sequence', sequence_trips), ('mixed', mixed_trips), ('random', random_trips)]:
        rou_file = os.path.join(output_dir, f"{mode}.rou.xml")
        
        cmd_duaro = [duarouter_path,
                     '-n', sumo_net_file,
                     '-t', trips_file,
                     '-a', vehicles_add_file,
                     '-o', rou_file,
                     '--ignore-errors']
        subprocess.run(cmd_duaro, check=True)
        
        # 清洗rou.xml，移除所有<vType>节点
        remove_vtype_from_rou(rou_file)
        
        rou_files[mode] = rou_file
        print(f"✅ 生成 {mode} 模式路由文件: {rou_file}")
    
    # 为所有rou.xml文件添加随机初始电量
    vehicle_config = get_battery_config()
    for mode, rou_file in rou_files.items():
        add_random_ev_charge_level(rou_file, vehicle_config)
        print(f"✅ 为 {mode} 模式添加随机初始电量")
    
    return rou_files

if __name__ == '__main__':
    # 参数设置
    vehicle_count = 10000
    ev_ratio = 0.18
    
    # 文件路径
    sumo_net_file = 'data/map/glasgow_clean.net.xml'
    output_dir = 'data/routes'
    vehicles_add_file = 'data/vehicles.add.xml'
    
    # SUMO工具路径
    sumo_home = os.environ.get('SUMO_HOME', '/usr/share/sumo')
    tools = os.path.join(sumo_home, 'tools')
    duarouter_path = 'duarouter'
    
    # 生成全局路由文件
    rou_files = generate_global_routes(vehicle_count, ev_ratio, sumo_net_file, tools, output_dir, vehicles_add_file, duarouter_path)
    
    print(f"\n🎉 全局路由文件生成完成！")
    for mode, rou_file in rou_files.items():
        print(f"   {mode}: {rou_file}") 
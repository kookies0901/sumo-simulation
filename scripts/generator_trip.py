import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from load_scenario import load_scenario

def remove_vtype_from_rou(rou_path):
    tree = ET.parse(rou_path)
    root = tree.getroot()
    vtypes = root.findall('vType')
    for vtype in vtypes:
        root.remove(vtype)
    tree.write(rou_path, encoding='utf-8', xml_declaration=True)

def generate_trip_routes(config: dict,
                         sumo_net_file: str,
                         output_dir: str,
                         sumo_tools_path: str,
                         vehicles_add_file: str = 'data/vehicles.add.xml',
                         duarouter_path: str = 'duarouter',
                         fringe_factor: int = 10) -> str:
    """
    Generate .rou.xml for a scenario using randomTrips.py + duarouter.
    Supports 'random' mode only. Generates separate trips for EV and petrol, merges, then routes.
    Vehicle types are defined externally in vehicles.add.xml.
    :param config: scenario configuration dict
    :param sumo_net_file: path to SUMO .net.xml file
    :param output_dir: directory to write trip/route files
    :param sumo_tools_path: path to SUMO tools directory containing randomTrips.py
    :param duarouter_path: path to duarouter executable
    :param fringe_factor: fringe factor to avoid dead-end roads
    :return: path to generated .rou.xml file
    """
    scenario_id = config['scenario_id']
    vehicle_count = config['vehicle_count']
    ev_ratio = config['ev_ratio']

    os.makedirs(output_dir, exist_ok=True)
    # File paths
    ev_trips = os.path.join(output_dir, f"{scenario_id}_ev.trips.xml")
    pet_trips = os.path.join(output_dir, f"{scenario_id}_pet.trips.xml")
    merged_trips = os.path.join(output_dir, f"{scenario_id}.trips.xml")
    rou_file = os.path.join(output_dir, f"{scenario_id}.rou.xml")

    # compute counts
    ev_count = int(vehicle_count * ev_ratio)
    pet_count = vehicle_count - ev_count

    # Base randomTrips command - restrict to passenger vehicles only
    base_cmd = [sys.executable, os.path.join(sumo_tools_path, 'randomTrips.py'),
                '-n', sumo_net_file,
                '--fringe-factor', str(fringe_factor),
                '--validate']  # Only use passenger vehicles, remove --validate to avoid unknown vehicle class errors

    # Generate EV trips - use individual trips instead of flows
    cmd_ev = base_cmd + [
        '-o', ev_trips,
        '-e', str(ev_count),  # Use -e for individual trips, not --flows
        '--prefix', 'EV_'
    ]
    subprocess.run(cmd_ev, check=True)

    # Generate petrol trips - use individual trips instead of flows
    cmd_pet = base_cmd + [
        '-o', pet_trips,
        '-e', str(pet_count),  # Use -e for individual trips, not --flows
        '--prefix', 'PET_'
    ]
    subprocess.run(cmd_pet, check=True)

    # Merge trip files into one trips.xml
    merge_trips_xml([ev_trips, pet_trips], merged_trips)

    # Run duarouter to generate .rou.xml (combine trips and external vehicle types)
    cmd_duaro = [duarouter_path,
                 '-n', sumo_net_file,
                 '-t', merged_trips,
                 '-a', vehicles_add_file,  # Include vehicle type definitions
                 '-o', rou_file,
                 '--ignore-errors']
    subprocess.run(cmd_duaro, check=True)

    # 清洗rou.xml，移除所有<vType>节点
    remove_vtype_from_rou(rou_file)

    return rou_file


def merge_trips_xml(input_files: list, output_file: str):
    """
    Merge multiple trip XML files into one for duarouter,
    and set vehicle type attribute based on ID prefix.
    Assumes vType definitions are provided externally.
    """
    # Create root <routes>
    routes = ET.Element('routes')

    for f in input_files:
        tree = ET.parse(f)
        for veh in tree.getroot():
            vid = veh.get('id', '')
            # Assign type attribute
            if vid.startswith('EV_'):
                veh.set('type', 'EV')
            elif vid.startswith('PET_'):
                veh.set('type', 'petrol')
            else:
                veh.set('type', 'unknown')
            routes.append(veh)

    # Write merged trips file without vType definitions
    ET.ElementTree(routes).write(output_file, xml_declaration=True, encoding='utf-8')


if __name__ == '__main__':
    cfg = load_scenario('S001', 'data/dataset_1/scenario_matrix.csv') # S001需要修改为实际的scenario_id
    net = os.path.join('data', 'map', 'glasgow_clean.net.xml')
    out = os.path.join('sumo', 'dataset_1', 'S001', 'routes') # S001需要修改为实际的scenario_id
    
    # Use system SUMO installation
    sumo_home = os.environ.get('SUMO_HOME', '/usr/share/sumo')
    tools = os.path.join(sumo_home, 'tools')
    duarouter_path = 'duarouter'  # System installed SUMO
    
    vehicles_add = os.path.join('data', 'vehicles.add.xml')
    rou = generate_trip_routes(cfg, net, out, tools, vehicles_add, duarouter_path)
    print(f"Generated route file at: {rou}")

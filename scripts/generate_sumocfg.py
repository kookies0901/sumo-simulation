import os
from load_scenario import load_scenario


def generate_sumocfg(scenario_id: str,
                      base_dir: str = "/home/ubuntu/project/MSC/Msc_Project/sumo/dataset_1",
                      net_file: str = "/home/ubuntu/project/MSC/Msc_Project/data/map/glasgow_clean.net.xml"):
    config = load_scenario(scenario_id)
    cs_layout_id = config["cs_layout_id"]

    scenario_dir = os.path.join(base_dir, scenario_id)
    route_path = os.path.join("routes", f"{scenario_id}.rou.xml")
    cs_path = os.path.join("cs", f"{cs_layout_id}.xml")
    sumocfg_path = os.path.join(scenario_dir, f"{scenario_id}.sumocfg")

    # 计算相对路径
    vehicles_add_path = os.path.relpath("data/vehicles.add.xml", scenario_dir)
    net_file_rel = os.path.relpath(net_file, scenario_dir)
    additional_files = f"{cs_path},{vehicles_add_path}"

    content = f"""<configuration>
    <input>
        <net-file value=\"{net_file_rel}\"/>
        <route-files value=\"{route_path}\"/>
        <additional-files value=\"{additional_files}\"/>
    </input>
    <time>
        <begin value=\"0\"/>
        <end value=\"3600\"/>
    </time>
</configuration>
"""

    os.makedirs(scenario_dir, exist_ok=True)
    with open(sumocfg_path, "w") as f:
        f.write(content)

    print(f"✅ SUMO config saved to {sumocfg_path}")


if __name__ == "__main__":
    generate_sumocfg("S001")    # S001需要修改为实际的scenario_id

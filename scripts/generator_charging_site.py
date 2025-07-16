import os
import json
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
            "power": "220.0",
            "efficiency": "0.95",
            "chargeDelay": "1.0",
            "chargingVehicles": ""
        })

    # Write output
    out_dir = os.path.join(out_dir_base, scenario_id, "cs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{cs_layout_id}.xml")

    tree = ET.ElementTree(root)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"✅ Charging stations for scenario '{scenario_id}' saved to: {out_path}")


if __name__ == "__main__":
    # Example usage:
    generate_charging_stations("S001")

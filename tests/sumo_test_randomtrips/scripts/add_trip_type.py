import xml.etree.ElementTree as ET
import random

input_file = "../sumo/generated_trips.trips.xml"
output_file = "../sumo/generated_trips_with_type.trips.xml"

tree = ET.parse(input_file)
root = tree.getroot()

for trip in root.findall("trip"):
    # 约90%为petrol，10%为EV
    trip_type = "petrol" if random.random() < 0.9 else "EV"
    trip.set("type", trip_type)

tree.write(output_file, encoding="utf-8", xml_declaration=True)
print(f"Written to {output_file}")

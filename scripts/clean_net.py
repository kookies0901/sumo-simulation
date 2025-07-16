import xml.etree.ElementTree as ET

def clean_net(input_file, output_file):
    illegal_classes = {"container", "drone", "subway", "wheelchair", "aircraft", "scooter", "cable_car"}
    tree = ET.parse(input_file)
    root = tree.getroot()
    count = 0
    for edge in root.findall('edge'):
        for lane in edge.findall('lane'):
            changed = False
            allow = lane.get('allow')
            disallow = lane.get('disallow')
            if allow and any(illegal in allow for illegal in illegal_classes):
                del lane.attrib['allow']
                changed = True
            if disallow and any(illegal in disallow for illegal in illegal_classes):
                del lane.attrib['disallow']
                changed = True
            if changed:
                count += 1
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"✅ Cleaned {count} lanes. Saved to {output_file}")

if __name__ == "__main__":
    input_file = 'data/map/glasgow.net.xml'
    output_file = 'data/map/glasgow_clean.net.xml'
    clean_net(input_file, output_file)

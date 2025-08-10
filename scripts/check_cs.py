# check_cs.py
import xml.etree.ElementTree as ET
net = ET.parse("data/map/glasgow_clean.net.xml").getroot()
cs  = ET.parse("data/cs/cs_group_001.xml").getroot()

lane_len = {}
for e in net.findall(".//lane"):
    lid = e.get("id"); L = float(e.get("length", "0")); lane_len[lid] = L

seen = set(); bad = []
for s in cs.findall(".//chargingStation"):
    cid = s.get("id"); lane = s.get("lane")
    pos = s.get("pos"); sPos = s.get("startPos"); ePos = s.get("endPos")
    if cid in seen: bad.append((cid, "DUPLICATE_ID")); 
    seen.add(cid)
    if (lane not in lane_len) or lane.startswith(":"):
        bad.append((cid, f"INVALID_LANE:{lane}")); continue
    L = lane_len[lane]
    if pos is not None:
        p = float(pos); 
        if not (0 <= p <= L): bad.append((cid, f"POS_OOB:{p}/{L}"))
    else:
        if sPos is None or ePos is None:
            bad.append((cid, "MISSING_START_END"))
        else:
            sp, ep = float(sPos), float(ePos)
            if not (0 <= sp <= L and 0 <= ep <= L and sp <= ep):
                bad.append((cid, f"RANGE_OOB:{sp}-{ep}/{L}"))
print("BAD:", len(bad))
for x in bad[:80]: print(x)

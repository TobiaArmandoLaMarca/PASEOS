import xml.etree.ElementTree as ET
import comtypes.client as cc
import re
import os

# === CONFIG ===
kml_path = r"C:\Users\LaMar\Downloads\S2B.kml"
save_path = r"C:\Users\LaMar\Documents\STK_Export\Sentinel_Areas.stk"

def safe_stk_name(name):
    return re.sub(r'[^\w\-]', '_', name)

# === PARSE KML ===
tree = ET.parse(kml_path)
root_kml = tree.getroot()
ns = {'kml': 'http://www.opengis.net/kml/2.2'}
placemarks = root_kml.findall(".//kml:Placemark", ns)

# === APRI STK ===
app = cc.CreateObject("STK11.Application")  # Cambia versione se STK12 o STK13
app.Visible = True
root = app.Personality2
root.NewScenario("Sentinel_Areas")
scenario = root.CurrentScenario
cmd = root.ExecuteCommand

# === CREA OGGETTI AREA TARGET ===
for placemark in placemarks:
    name_raw = placemark.find("kml:name", ns).text
    name = safe_stk_name(name_raw)

    # Coordinate
    coords_elem = placemark.find(".//kml:coordinates", ns)
    if coords_elem is None:
        continue
    coords_text = coords_elem.text.strip().split()
    coords = [tuple(map(float, c.split(','))) for c in coords_text]
    if len(coords) < 3:
        continue

    # Costruisci lista LON LAT piatta
    flat_coords = []
    for lon, lat, *_ in coords:
        flat_coords.append(f"{lon:.6f}")
        flat_coords.append(f"{lat:.6f}")

    # Crea AreaTarget
    try:
        cmd(f'Delete / */AreaTarget/{name}')
    except:
        pass
    cmd(f'New / */AreaTarget "{name}"')
    cmd(f'SetPattern */AreaTarget/{name} LatLon {len(flat_coords)//2} {" ".join(flat_coords)}')

    # TimeSpan
    timespan = placemark.find("kml:TimeSpan", ns)
    if timespan is not None:
        begin = timespan.find("kml:begin", ns)
        end = timespan.find("kml:end", ns)
        if begin is not None and end is not None:
            begin_str = begin.text.replace("T", " ").replace("Z", "")
            end_str = end.text.replace("T", " ").replace("Z", "")
            cmd(f'AccessConstraints */AreaTarget/{name} TimeInterval Set "{begin_str}" "{end_str}"')

# === SALVA LO SCENARIO ===
if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))
root.SaveAs(save_path)

print(f"\n✅ Scenario salvato come:\n{save_path}")








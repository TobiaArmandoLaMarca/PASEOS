import os
import getpass
import xml.etree.ElementTree as ET
from datetime import datetime
from win32com.client import Dispatch
from dateutil import parser  # pip install python-dateutil

KML_FILENAME = "S2B.kml"

def get_kml_path():
    user = getpass.getuser()
    download_dir = os.path.join("C:\\Users", user, "Downloads")
    path = os.path.join(download_dir, KML_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File '{KML_FILENAME}' non trovato nella cartella Download.")
    return path

def convert_time_to_stk_format(time_str):
    try:
        dt = parser.isoparse(time_str.strip())
        return dt.strftime("%d %b %Y %H:%M:%S.000")
    except Exception as e:
        raise ValueError(f"⚠️ Errore nella conversione del tempo '{time_str}': {e}")

def parse_kml(file_path):
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    tree = ET.parse(file_path)
    root = tree.getroot()
    placemarks = root.findall(".//kml:Placemark", ns)

    areas = []
    for placemark in placemarks:
        name_el = placemark.find("kml:name", ns)
        name = name_el.text.strip() if name_el is not None else "Unnamed"

        coords_el = placemark.find(".//kml:coordinates", ns)
        if coords_el is None:
            raise ValueError(f"❌ Placemark '{name}' senza coordinate.")
        coords_text = coords_el.text.strip().split()

        coordinates = []
        for c in coords_text:
            lon_deg = float(c.split(",")[0])
            lat_deg = float(c.split(",")[1])

            original_lon = lon_deg  # Salva per confronto

            # 🔧 Correzione longitudini fuori range
            if lon_deg < -180:
                lon_deg += 360
                print(f"🔁 Longitudine normalizzata: {original_lon}° → {lon_deg}°")
            elif lon_deg > 180:
                lon_deg -= 360
                print(f"🔁 Longitudine normalizzata: {original_lon}° → {lon_deg}°")

            coordinates.append((lat_deg, lon_deg))

        if coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])

        begin_time = None
        end_time = None
        time_span = placemark.find("kml:TimeSpan", ns)
        if time_span is not None:
            begin_el = time_span.find("kml:begin", ns)
            end_el = time_span.find("kml:end", ns)
            if begin_el is not None and end_el is not None:
                begin_time = convert_time_to_stk_format(begin_el.text.strip())
                end_time = convert_time_to_stk_format(end_el.text.strip())

        areas.append({
            'name': name,
            'coordinates': coordinates,
            'begin': begin_time,
            'end': end_time
        })

    return areas

def create_scenario_and_objects(areas, scenario_name="KML_Area_Import", satellite_name="MySatellite"):
    app = Dispatch("STK11.Application")
    app.Visible = True
    root = app.Personality2

    print("📡 Creo nuovo scenario...")
    root.NewScenario(scenario_name)
    root.UnitPreferences.SetCurrentUnit("DateFormat", "UTCG")
    scenario = root.CurrentScenario

    print(f"🛰️ Creo satellite dummy: {satellite_name}")
    satellite = scenario.Children.New(18, satellite_name)  # 18 = eSatellite

    for idx, area in enumerate(areas):
        unique_name = f"{area['name']}_{idx}"
        print(f"\n📍 Creo AreaTarget: {unique_name} ({len(area['coordinates'])} punti)")
        try:
            area_target = scenario.Children.New(2, unique_name)  # 2 = eAreaTarget

            root.BeginUpdate()
            area_target.AreaType = 1  # ePattern
            patterns = area_target.AreaTypeData
            patterns.RemoveAll()

            for lat, lon in area['coordinates']:
                patterns.Add(lat, lon)

            area_target.AutoCentroid = True
            root.EndUpdate()
            print(f"✅ {unique_name} creato correttamente.")

            # Accesso satellite → area_target
            access = satellite.GetAccessToObject(area_target)

            if area.get("begin") and area.get("end"):
                access.AccessTimePeriod = 2  # eUserSpecAccessTime
                access.AccessTimePeriodData.AccessInterval.SetExplicitInterval(area["begin"], area["end"])
                print(f"🕒 Intervallo accesso definito: {area['begin']} - {area['end']}")
            else:
                print("ℹ️ Nessun intervallo temporale definito: accesso non limitato.")

        except Exception as e:
            raise RuntimeError(
                f"\n⛔ Errore nella creazione dell'area '{unique_name}'\n"
                f"🔻 Coordinate:\n" +
                "\n".join([f"   [{i+1}] Lat: {lat}, Lon: {lon}" for i, (lat, lon) in enumerate(area['coordinates'])]) +
                f"\n❌ Eccezione sollevata: {e}"
            )

    # Salvataggio scenario
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario_filename = f"auto_saved_{timestamp}.sc"
    scenario_dir = os.path.join(os.path.expanduser("~"), "Downloads", "STK_Scenarios")
    os.makedirs(scenario_dir, exist_ok=True)
    scenario_path = os.path.join(scenario_dir, scenario_filename)

    try:
        root.SaveAs(scenario_path)
        print(f"\n💾 Scenario salvato in: {scenario_path}")
    except Exception as save_err:
        raise RuntimeError(f"\n❌ Errore nel salvataggio dello scenario:\n{save_err}")

def main():
    try:
        print("🔍 Caricamento file KML...")
        kml_path = get_kml_path()
        areas = parse_kml(kml_path)

        if not areas:
            raise ValueError("⚠️ Nessuna area trovata nel file.")

        print(f"📦 {len(areas)} area target trovate.")
        create_scenario_and_objects(areas)
        print("\n🎯 Completato. Apri STK e calcola accessi da interfaccia.")

    except Exception as ex:
        print(f"\n🚨 Script interrotto per errore:\n{ex}")
        raise

if __name__ == "__main__":
    main()



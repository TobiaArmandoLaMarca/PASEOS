import numpy as np
import pymap3d as pm
import pykep as pk
from paseos.actors.actor_builder import ActorBuilder
from paseos.actors.spacecraft_actor import SpacecraftActor
from paseos.attitude.attitude import Attitude
from astropy.time import Time
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta, timezone
from paseos.observation.EarthObservation import EOTools 
import time
import paseos  # Importa PASEOS correttamente

# Definizione del satellite Kanyini

sat_actor = ActorBuilder.get_actor_scaffold(name="Kanyini",actor_type=SpacecraftActor,epoch= pk.epoch_from_string("2025-03-05 19:00:28"))
earth = pk.planet.jpl_lp("earth")

ActorBuilder.set_orbit(actor=sat_actor,
    position=[-5529688.37, 4081251.929, 1308.67035],
    velocity=[593.22742, 783.20028099, 7553.094223426],
    epoch= pk.epoch_from_string("2025-03-05 19:00:28"),  # Sincronizza con l'epoca iniziale
    central_body=earth
)

t0 = pk.epoch_from_string("2025-03-05 19:00:28")  # Tempo iniziale
# Definizione assetto e sensore
cfg = paseos.load_default_cfg()
cfg.sim.time_multiplier = 25.0  # Acceleriamo la simulazione di 100x
cfg.sim.start_time = t0.mjd2000 * 86400  # Convertiamo MJD2000 in secondi

# Sincronizziamo manualmente l'epoca dell'attore
#sat_actor._local_time = t0

paseos_instance = paseos.init_sim(sat_actor, cfg)  # Avvia la simulazione

sat_attitude = Attitude(sat_actor)
eo_tools = EOTools(
    local_actor=sat_actor,
    actor_initial_attitude_in_deg=[0.0, 0.0, 0.0],
    actor_FOV_ACT_in_deg=[20.0],
    actor_FOV_ALT_in_deg=[20.0],
    actor_pointing_vector_body=[0.0, 0.0, 1.0]
)

# Caricamento footprint Sentinel
sentinel_kml = eo_tools.load_kml(r"C:\Users\LaMar\miniforge3\envs\esaenv\Lib\site-packages\paseos\Sentinel2B.kml")

# Tempo finale della simulazione
t_final = pk.epoch(t0.mjd2000 + 7.0)  # Aggiunge 7 giorni

# Lista per registrare le epoche di intersezione
intersections_data = []

# ✅ Step di avanzamento di 100 secondi simulati
time_step = 100

# ✅ Loop per avanzare la simulazione a step di 100 secondi simulati
while sat_actor.local_time.mjd2000 <= t_final.mjd2000:
    # Avanza la simulazione di 100 secondi simulati
    paseos_instance.advance_time(time_to_advance=time_step, current_power_consumption_in_W=0)

    # Otteniamo il tempo attuale aggiornato
    current_t = sat_actor.local_time
    datetime_utc = (datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(days=current_t.mjd2000)).replace(tzinfo=None)

    # Posizione e velocità aggiornate
    r = np.array(sat_actor.get_position(current_t))
    v = np.array(sat_actor.get_position_velocity(current_t)[1])
    # Calcola footprint del sensore
    ray_directions = eo_tools.get_fov_vectors_in_BRF()
    eul_ang = [0.0, 0.0, 0.0]
    intersections_matrix = eo_tools._find_intersection_in_Geodetic(ray_directions, eul_ang, datetime_utc, r, v)

    FovPoints = np.array([
        [intersections_matrix[0, 0], intersections_matrix[1, 0]],
        [intersections_matrix[0, 1], intersections_matrix[1, 1]],
        [intersections_matrix[0, 2], intersections_matrix[1, 2]],
        [intersections_matrix[0, 3], intersections_matrix[1, 3]]
    ])
    
    # ✅ Controlla se il FOV di Kanyini interseca Sentinel
    check_results = EOTools.check_fov_in_polygon(sentinel_kml, datetime_utc, FovPoints)
    
    #import pdb; pdb.set_trace()
    if 'fov_inside' in check_results and check_results['fov_inside']:
        intersections_data.append((datetime_utc, check_results['coverage_ratio']))  # ✅ Salviamo solo se c'è un'intersezione

# ✅ Salviamo le epoche di intersezione in un file solo alla fine
with open("intersections_data.txt", "w") as f:
    for epoch, coverage in intersections_data:  # Ora iteriamo su (epoca, coverage)
        f.write(f"{epoch.isoformat()} - Coverage: {coverage:.2f}%\n")

print(f"Totale intersezioni trovate: {len(intersections_data)}")

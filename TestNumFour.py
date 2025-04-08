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

# Definiamo il satellite
sat_actor = ActorBuilder.get_actor_scaffold(name="mySat", actor_type=SpacecraftActor, epoch=pk.epoch_from_string("2026-01-01 12:00:00"))
earth = pk.planet.jpl_lp("earth")
ActorBuilder.set_orbit(actor=sat_actor,
                       position=[1616740, 6892990, 0],
                       velocity=[940.851, -220.658, 7440],
                       epoch=pk.epoch_from_string("2026-01-01 12:00:00"), central_body=earth)

# Creiamo l'istanza Attitude
sat_attitude = Attitude(sat_actor)

# Crea l'istanza di EOTools
eo_tools = EOTools(
    local_actor=sat_actor,
    actor_initial_attitude_in_deg=[0.0, 0.0, 0.0],
    actor_FOV_ACT_in_deg=[20.0],
    actor_FOV_ALT_in_deg=[20.0],
    actor_pointing_vector_body=[0.0, 0.0, 1.0],
)

import numpy as np
import pykep as pk
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta, timezone
import time
from paseos.observation.EarthObservation import EOTools

# Attiva modalità interattiva
plt.ion()

# Creiamo la figura e l'asse per la mappa
fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})

# Imposta la mappa con elementi stabili
ax.set_title("Traccia a terra del satellite e footprint del sensore")
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

# Traccia a terra
trace_lons = []
trace_lats = []

# Parametri di simulazione
time_step = 100  # secondi
num_steps = 10  # passi temporali
t0 = pk.epoch_from_string("2026-01-01 12:00:00.000")
epoch_j2000 = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
t0_seconds_since_j2000 = t0.mjd2000 * 86400

# Simulazione temporale
for step in range(num_steps):
    try:
        t = t0_seconds_since_j2000 + step * time_step
        datetime_utc = epoch_j2000 + timedelta(seconds=t)
        datetime_utc = datetime_utc.replace(tzinfo=None)

        # Convertire in `pykep.epoch`
        epoch_final = pk.epoch_from_string(datetime_utc.strftime("%Y-%b-%d %H:%M:%S"))

        # Ottenere posizione e velocità del satellite
        r = np.array(sat_actor.get_position(epoch_final))
        v = np.array(sat_actor.get_position_velocity(epoch_final)[1])

        # Calcola i vettori del FOV
        ray_directions = eo_tools.get_fov_vectors_in_BRF()
        eul_ang = [0.0, 0.0, 0.0]

        # Calcola intersezioni
        intersections_matrix = eo_tools._find_intersection_in_Geodetic(ray_directions, eul_ang, datetime_utc, r, v)

        # Estrai coordinate lat/lon
        FovPoints = np.array([
            [intersections_matrix[0, 0], intersections_matrix[1, 0]],
            [intersections_matrix[0, 1], intersections_matrix[1, 1]],
            [intersections_matrix[0, 2], intersections_matrix[1, 2]],
            [intersections_matrix[0, 3], intersections_matrix[1, 3]]
        ])

        # Aggiorna la traccia a terra
        trace_lons.append(intersections_matrix[0, 0])
        trace_lats.append(intersections_matrix[1, 0])

        # **Plotta la traccia a terra SENZA CANCELLARE GLI ELEMENTI FISSI**
        ax.plot(trace_lons, trace_lats, linestyle='-', color='red', transform=ccrs.PlateCarree())

        # **Plotta il footprint senza punti rossi**
        eo_tools.plot_fov_on_map(FovPoints, ax)

        # **Aggiorna il grafico senza scatti**
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        #time.sleep(0.1)

    except ValueError as e:
        print(f"Errore a step {step}:", e)

# Manteniamo la figura aperta alla fine dell'animazione
plt.ioff()
plt.show()


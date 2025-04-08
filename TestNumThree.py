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
#sim = paseos.init_sim(sat_actor)
# Parametri per la simulazione
time_step = 520  # secondi
num_steps = 14  # 1 ora di simulazione con step di 100 secondi
datetime_utc = datetime(2026, 1, 1, 12, 0, 0)

# Creiamo la figura per la mappa
plt.figure()
fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.set_title("Traccia a terra del satellite e footprint del sensore")
# Inizializza la lista per tracciare la traccia a terra
trace_lons = []
trace_lats = []
t0 = pk.epoch_from_string("2026-01-01 12:00:00.000")
epoch_j2000 = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
t0_seconds_since_j2000 = t0.mjd2000*86400
# Simulazione temporale
for step in range(num_steps):
    try:
        t = (t0_seconds_since_j2000 + step*time_step)
        datetime_utc = epoch_j2000 + timedelta(seconds=t)
        print("MJD2000 di t0:", t0.mjd2000) 
        print("Secondi da J2000:", t0_seconds_since_j2000) 
        datetime_utc = datetime_utc.replace(tzinfo=None)
        #import pdb; pdb.set_trace()
        #epoch_time = Time(datetime_utc, format='datetime', scale='utc')

        #datetime_utc = datetime(datetime_utc)
        #import pdb; pdb.set_trace()
        # Calcola i vettori del FOV nel BRF
        ray_directions = eo_tools.get_fov_vectors_in_BRF()
        eul_ang = [0.0, 0.0, 0.0]

        # ✅ Convertire datetime_utc in un oggetto `pykep.epoch`
        epoch_final = pk.epoch_from_string(datetime_utc.strftime("%Y-%b-%d %H:%M:%S"))

        # ✅ Ora possiamo passarlo a PASEOS senza errori
        r = np.array(sat_actor.get_position(epoch_final))
        v = np.array(sat_actor.get_position_velocity(epoch_final)[1])
        # Ora passiamo `epoch_final` a PASEOS
        #r = np.array(sat_actor.get_position(sat_actor.local_time))
        #v = np.array(sat_actor.get_position_velocity(sat_actor.local_time)[1])
        #r = np.array(sat_actor.get_position(sat_actor.local_time))
        #v = np.array(sat_actor.get_position_velocity(sat_actor.local_time))
        intersections_matrix = eo_tools._find_intersection_in_Geodetic(ray_directions, eul_ang, datetime_utc,r,v)
        
        # Estrai le coordinate dei punti di intersezione (latitudine, longitudine)
        FovPoints = np.array([
            [intersections_matrix[0, 0], intersections_matrix[1, 0]],
            [intersections_matrix[0, 1], intersections_matrix[1, 1]],
            [intersections_matrix[0, 2], intersections_matrix[1, 2]],
            [intersections_matrix[0, 3], intersections_matrix[1, 3]]
        ])
        
        # Aggiorna la traccia a terra del satellite
        trace_lons.append(intersections_matrix[0, 0])
        trace_lats.append(intersections_matrix[1, 0])
        
        # Plotta la traccia a terra e la footprint senza cancellare
        #ax.plot(trace_lons, trace_lats, linestyle='-', color='red', label="Traccia a terra")
        eo_tools.plot_fov_on_map(FovPoints, ax)
        
        # Aggiorna il grafico senza cancellarlo
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.1)
    except ValueError as e:
        print(f"Errore a step {step}:", e)
    
    # Aggiorna il tempo
    #t += timedelta(seconds=time_step)

plt.show()
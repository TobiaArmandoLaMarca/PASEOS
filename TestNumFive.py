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


# Supponiamo di avere un attore (satellite) con posizione in orbita
sat_actor = ActorBuilder.get_actor_scaffold(name="mySat", actor_type=SpacecraftActor, epoch=pk.epoch_from_string("2025-03-05 19:00:28"))
earth = pk.planet.jpl_lp("earth")
ActorBuilder.set_orbit(actor=sat_actor,
                       position=[-5529688.37, 4081251.929, 1308.67035],
                       velocity=[593.22742, 783.20028099, 7553.094223426],
                       epoch=pk.epoch_from_string("2025-03-05 19:00:28"), central_body=earth)

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
trace_lons = []
trace_lats = []
t0 = pk.epoch_from_string("2025-03-06 15:54:28.000")
epoch_j2000 = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
t0_seconds_since_j2000 = t0.mjd2000*86400
t = (t0_seconds_since_j2000)
datetime_utc = epoch_j2000 + timedelta(seconds=t)
print("MJD2000 di t0:", t0.mjd2000) 
print("Secondi da J2000:", t0_seconds_since_j2000) 

datetime_utc = datetime_utc.replace(tzinfo=None)
print(datetime_utc)
ray_directions = eo_tools.get_fov_vectors_in_BRF()
eul_ang = [0.0, 0.0, 0.0]

epoch_final = pk.epoch_from_string(datetime_utc.strftime("%Y-%b-%d %H:%M:%S"))

r = np.array(sat_actor.get_position(epoch_final))
v = np.array(sat_actor.get_position_velocity(epoch_final)[1])

intersections_matrix = eo_tools._find_intersection_in_Geodetic(ray_directions, eul_ang, datetime_utc,r,v)
        
FovPoints = np.array([
[intersections_matrix[0, 0], intersections_matrix[1, 0]],
[intersections_matrix[0, 1], intersections_matrix[1, 1]],
[intersections_matrix[0, 2], intersections_matrix[1, 2]],
[intersections_matrix[0, 3], intersections_matrix[1, 3]]
])

# Carica il file KML di Sentinel
sentinel_kml = eo_tools.load_kml(r"C:\Users\LaMar\miniforge3\envs\esaenv\Lib\site-packages\paseos\Sentinel2B.kml")

# Ottieni il footprint di Sentinel più vicino alla simulation_time
check_results = EOTools.check_fov_in_polygon(sentinel_kml, datetime_utc, FovPoints)
print(check_results)  # Stampa i dettagli

sentinel_footprint = check_results['selected_polygon']

# Estrarre solo (lon, lat) ignorando la terza coordinata (h)
sentinel_coords = list(sentinel_footprint.exterior.coords)
sentinel_lons, sentinel_lats = zip(*[(lon, lat) for lon, lat, *_ in sentinel_coords])  # Ignora h
#import pdb; pdb.set_trace()
# Creazione di UNA SOLA mappa
plt.figure()
fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})

# Aggiungere feature di base
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.set_title("Footprint del sensore e Sentinel")

# Plottare il FOV del tuo satellite
eo_tools.plot_fov_on_map(FovPoints, ax)

# Plottare il FOV di Sentinel
#ax.plot(sentinel_lons, sentinel_lats, color='blue', linestyle='-', label="Sentinel FOV", transform=ccrs.PlateCarree())
#ax.fill(sentinel_lons, sentinel_lats, color='blue', alpha=0.3, transform=ccrs.PlateCarree())
ax.scatter(sentinel_lons, sentinel_lats, color="red", s=10, transform=ccrs.PlateCarree())  # Plotta i punti
ax.plot(sentinel_lons, sentinel_lats, color='blue', linestyle='-', label="Sentinel FOV", transform=ccrs.PlateCarree())
ax.fill(sentinel_lons, sentinel_lats, color='blue', alpha=0.3, transform=ccrs.PlateCarree())


# Legenda
ax.legend()

# Mostrare il grafico
plt.show()
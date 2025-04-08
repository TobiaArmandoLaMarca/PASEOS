import numpy as np
import pymap3d as pm
import pykep as pk
#import paseos
from paseos.actors.actor_builder import ActorBuilder
from paseos.actors.spacecraft_actor import SpacecraftActor
from paseos.attitude.attitude import Attitude
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import contextily as ctx
from datetime import datetime
from paseos.observation.EarthObservation import EOTools 
from PIL import Image

# Supponiamo di avere un attore (satellite) con posizione in orbita
sat_actor = ActorBuilder.get_actor_scaffold(name="mySat", actor_type=SpacecraftActor, epoch=pk.epoch_from_string("2026-01-01 12:00:00"))
earth = pk.planet.jpl_lp("earth")
ActorBuilder.set_orbit(actor=sat_actor,
                       position=[1616740, 6892990, 0],
                       velocity=[940.851, -220.658, 7440],
                       epoch=pk.epoch_from_string("2026-01-01 12:00:00"), central_body=earth)

# Creiamo l'istanza Attitude
sat_attitude = Attitude(sat_actor),



# Crea l'istanza di EOTools
eo_tools = EOTools(
    local_actor=sat_actor,
    actor_initial_attitude_in_deg= [0.0, 0.0, 0.0],
    actor_FOV_ACT_in_deg=[20.0],
    actor_FOV_ALT_in_deg=[20.0],
    actor_pointing_vector_body= [0.0, 0.0, 1.0],
)

# Ottengo i vettori del FOV in BRF
ray_directions = eo_tools.get_fov_vectors_in_BRF()

#Definisco gli angoli di Eulero per la trasformazione
eul_ang = [0.0, 0.0, 0.0]
datetime_utc = datetime(2026, 1, 1, 12, 0, 0)
r = np.array(sat_actor.get_position(sat_actor.local_time))
v = np.array(sat_actor.get_position_velocity(sat_actor.local_time)[1])
time = datetime_utc
#import pdb; pdb.set_trace()
try:
    intersections_matrix = eo_tools._find_intersection_in_Geodetic(ray_directions, eul_ang,time,r,v)
    print("Punti di intersezione in ECEF (matrice 3x4):")
    print(intersections_matrix)
except ValueError as e:
    print("Errore:", e)


# Creiamo la figura e l'asse con Cartopy
fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})

# Esempio di punti di intersezione del FOV
FovPoints = np.array([
    [intersections_matrix[0,0], intersections_matrix[1,0]],  # Punto 1
    [intersections_matrix[0,1], intersections_matrix[1,1]],  # Punto 2
    [intersections_matrix[0,2], intersections_matrix[1,2]],  # Punto 3
    [intersections_matrix[0,3], intersections_matrix[1,3]]  # Punto 4
])
#import pdb; pdb.set_trace()
# Chiamiamo la funzione per plottare il FOV
eo_tools.plot_fov_on_map(FovPoints, ax)

# Mostra la figura finale
plt.show()

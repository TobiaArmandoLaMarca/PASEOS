import numpy as np
import pymap3d as pm
import pykep as pk
#import paseos 
from paseos.actors.actor_builder import ActorBuilder
from paseos.actors.spacecraft_actor import SpacecraftActor
from paseos.attitude.attitude import Attitude
from datetime import datetime

import sys
print("Python usato:", sys.executable)

# Supponiamo di avere un attore (satellite) con posizione in orbita
sat_actor = ActorBuilder.get_actor_scaffold(name="mySat", actor_type=SpacecraftActor,epoch = pk.epoch(0))
earth = pk.planet.jpl_lp("earth")
# Definiamo l'orbita del satellite
ActorBuilder.set_orbit(actor=sat_actor,
                       position = [1527667.32, 6513242.8, 0],
                       velocity = [967.89, -227, 7654],
                       epoch=pk.epoch(0), central_body=earth)


# Creiamo l'istanza Attitude
sat_attitude = Attitude(sat_actor)

# Definizione del target sulla Terra (Abu Dhabi)
lat_target = 30  # gradi
lon_target = -28  # gradi
alt_target = 0.0        # metri (livello del mare)

# Tempo di osservazione
datetime_utc = datetime(2026, 1, 1, 12, 0, 0)

# Definiamo gli angoli di Eulero (esempio generico)
eul_ang = np.array([0.0, 0.0, 0.0])  # Nessuna rotazione

# Calcolo dell'angolo di osservazione
angle = sat_attitude.compute_obs_angle(lat_target, lon_target, alt_target, datetime_utc, eul_ang)

print(f"Angolo di osservazione tra asse Z body e target: {angle:.2f}°")
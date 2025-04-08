import sys
sys.path.append("..")
sys.path.append("../..")
import paseos
from paseos import ActorBuilder, SpacecraftActor
import pykep as pk
import matplotlib.pyplot as plt
import orekit
import math
from orekit.pyhelpers import setup_orekit_curdir
from orekit_propagator import OrekitPropagator
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from orekit.pyhelpers import absolutedate_to_datetime
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

# Inizializza la VM Orekit e setup della cartella dati
vm = orekit.initVM()
setup_orekit_curdir()

# Epoca iniziale
utc = TimeScalesFactory.getUTC()
epoch = AbsoluteDate(2025, 3, 5, 19, 0, 28.000, utc)

# Parametri orbitali e satellite
orbital_elements = [
    6874713.599100338,
    0.001444270641378492,
    math.radians(97.4108605048587),
    math.radians(78.42008002143851),
    math.radians(143.571800002229),
    math.radians(281.5909218918905)
]

satellite_mass = 9.54
area_d = 0.164
area_s = 0.162
cr_s = 1.5
cd = 2.2

# Inizializza il propagatore
propagator = OrekitPropagator(orbital_elements, epoch, satellite_mass, area_s, cr_s, area_d, cd)

# Parametri temporali di propagazione
duration_sec = 1814400.0  # 21 giorni
step_sec = 10.0

import os

# Parametri temporali di propagazione
duration_sec = 1814400.0  # 21 giorni
step_sec = 10.0
t_elapsed = 0.0

# Accesso diretto al propagatore numerico
propagator = OrekitPropagator(orbital_elements, epoch, satellite_mass,area_s,cr_s,area_d,cd)

# Stato iniziale
state = propagator.getInitialState()

# File di output
output_file = "propagation_output.txt"
with open(output_file, "w") as f:
    f.write("date, x (m), y (m), z (m), vx (m/s), vy (m/s), vz (m/s)\n")

    while t_elapsed <= duration_sec:
        # Propaga di uno step
        new_date = state.getDate().shiftedBy(step_sec)
        new_state = propagator.propagate(new_date)

        # Estrai PV
        pv = new_state.getPVCoordinates()
        pos = pv.getPosition()
        vel = pv.getVelocity()
        date_str = absolutedate_to_datetime(new_date).isoformat()

        # Coordinate
        x, y, z = pos.getX(), pos.getY(), pos.getZ()
        vx, vy, vz = vel.getX(), vel.getY(), vel.getZ()

        # Log
        print(f"[{date_str}] Pos: ({x:.2f}, {y:.2f}, {z:.2f}) m | Vel: ({vx:.2f}, {vy:.2f}, {vz:.2f}) m/s")
        f.write(f"{date_str}, {x:.3f}, {y:.3f}, {z:.3f}, {vx:.6f}, {vy:.6f}, {vz:.6f}\n")

        # Avanza
        state = new_state
        numerical.setInitialState(state)
        t_elapsed += step_sec

print(f"\n✅ Dati salvati in {os.path.abspath(output_file)}")

import sys
import math
import numpy as np
import matplotlib.pyplot as plt

import orekit
from orekit.pyhelpers import setup_orekit_curdir, absolutedate_to_datetime
from orekit_propagator import OrekitPropagator
from org.orekit.time import AbsoluteDate, TimeScalesFactory

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

# Esegui propagazione
results = propagator.propagate_at_fixed_times(duration_sec=duration_sec, step_sec=step_sec)

# Estrai dati
time_list = [entry['date'].durationFrom(epoch) for entry in results]
densities = [entry['density'] for entry in results]
A = [entry['a'] for entry in results]
I = [np.degrees(entry['i']) for entry in results]

# Estrai e salva vettori di stato (r, v)
state_vectors = []
for entry in results:
    r = entry['position']
    v = entry['velocity']
    state_vectors.append([
        r.getX(), r.getY(), r.getZ(),
        v.getX(), v.getY(), v.getZ()
    ])
state_vectors = np.array(state_vectors)

# Salva su file
np.savetxt("density_profile.txt", densities, header="Density [kg/m^3]", fmt="%.6e")
np.savetxt("state_vector.txt", state_vectors,
           header="x [m], y [m], z [m], vx [m/s], vy [m/s], vz [m/s]",
           fmt="%.6e", delimiter=", ")

# Plot densità
plt.figure()
plt.plot(time_list, densities)
plt.xlabel("Time [s]")
plt.ylabel("Density [kg/m^3]")
plt.yscale("log")
plt.title("Atmospheric Density over Time")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot semiasse maggiore
plt.figure()
plt.plot(time_list, A)
plt.xlabel("Time [s]")
plt.ylabel("Semi-major axis [m]")
plt.title("Semi-major Axis over Time")
plt.grid(True)
plt.tight_layout()
plt.show()

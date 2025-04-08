import sys
import math
import numpy as np
import matplotlib.pyplot as plt

import orekit
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.data import DataProvidersManager, DirectoryCrawler
from java.io import File

from orekit.pyhelpers import absolutedate_to_datetime
from orekit_propagator import OrekitPropagator

# =============================
# 🚀 Inizializza Orekit e carica dati
# =============================
vm = orekit.initVM()

# Specifica il path alla cartella "orekit-data"
orekit_data_path = File("C:/Users/LaMar/orekit-data")  # <-- Modifica qui se serve
manager = DataProvidersManager.getStaticInstance()
manager.clearProviders()
manager.addProvider(DirectoryCrawler(orekit_data_path))

# =============================
# 🕒 Definisci epoca iniziale
# =============================
utc = TimeScalesFactory.getUTC()
epoch = AbsoluteDate(2025, 3, 5, 19, 0, 28.000, utc)

# =============================
# 🚀 Parametri orbitali e satellite
# =============================
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

# =============================
# 🛰️  Inizializza propagatore
# =============================
propagator = OrekitPropagator(orbital_elements, epoch, satellite_mass, area_s, cr_s, area_d, cd)

duration_sec = 1814400.0
step_sec = 10.0

results = propagator.propagate_at_fixed_times(duration_sec=duration_sec, step_sec=step_sec)

# =============================
# 📊 Estrai dati
# =============================
r = [entry['position'] for entry in results]
v = [entry['velocity'] for entry in results]
A = [entry['a'] for entry in results]
I = [np.degrees(entry['i']) for entry in results]
time_list = [entry['date'].durationFrom(epoch) for entry in results]

# =============================
# 💾 Salva su file
# =============================
data = np.array([[ri.getX(), ri.getY(), ri.getZ(),
                  vi.getX(), vi.getY(), vi.getZ()] for ri, vi in zip(r, v)])

np.savetxt("state_vectors.txt", data,
           header="X[m] Y[m] Z[m] VX[m/s] VY[m/s] VZ[m/s]",
           fmt="%.6f")

# =============================
# 📈 Plot orbita e velocità
# =============================
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot([i.getX() for i in r], [i.getY() for i in r])
ax[0].set_xlabel("X [m]")
ax[0].set_ylabel("Y [m]")
ax[0].set_title("Orbit")
ax[0].set_aspect("equal", "box")

ax[1].plot([i.getNorm() for i in v])
ax[1].set_xlabel("Time [s]")
ax[1].set_ylabel("Velocity [m/s]")
ax[1].set_title("Velocity magnitude")
fig.tight_layout()
plt.show()

# =============================
# 📈 Plot inclinazione
# =============================
plt.figure()
plt.plot(time_list, I)
plt.xlabel("Time [s]")
plt.ylabel("Inclination [deg]")
plt.title("Inclination over time")
plt.grid(True)
plt.tight_layout()
plt.show()

# =============================
# 📈 Plot semi-asse maggiore
# =============================
plt.figure()
plt.plot(time_list, A)
plt.xlabel("Time [s]")
plt.ylabel("Semi-major axis")
plt.title("Semi-major axis over time")
plt.grid(True)
plt.tight_layout()
plt.show()


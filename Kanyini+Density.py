import sys
sys.path.append("..")
sys.path.append("../..")

# Import standard
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone

# Orekit setup
import orekit
from orekit.pyhelpers import setup_orekit_curdir, absolutedate_to_datetime
from orekit_propagator import OrekitPropagator
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.models.earth.atmosphere import NRLMSISE00
from org.orekit.models.earth.atmosphere.data import CssiSpaceWeatherData
from org.orekit.data import DataContext, DirectoryCrawler
from java.io import File

# Init VM e setup orekit
vm = orekit.initVM()
setup_orekit_curdir()

# Tempo e orbita iniziale
utc = TimeScalesFactory.getUTC()
epoch = AbsoluteDate(2025, 3, 5, 19, 00, 28.000, utc)

orbital_elements = [6874713.599100338, 0.001444270641378492, math.radians(97.4108605048587),
                    math.radians(78.42008002143851), math.radians(143.571800002229), math.radians(281.5909218918905)]
satellite_mass = 9.54
area_d = 0.164
area_s = 0.162
cr_s = 1.5
cd = 2.2

# Propagatore
propagator = OrekitPropagator(orbital_elements, epoch, satellite_mass, area_s, cr_s, area_d, cd)

# Setup modello NRLMSISE00 con dati CSSI
orekitData = File("C:/Users/LaMar/miniforge3/envs/esaenv/Lib/site-packages/paseos/orekit-data/CSSI-Space-Weather-Data")
manager = DataContext.getDefault().getDataProvidersManager()
manager.clearProviders()
manager.addProvider(DirectoryCrawler(orekitData))

cssi = CssiSpaceWeatherData(CssiSpaceWeatherData.DEFAULT_SUPPORTED_NAMES, manager, utc)
atmosphere = NRLMSISE00(cssi, propagator.sun, propagator.wgs84_ellipsoid)

# Propagazione
duration_sec = 1814400.0  # 21 giorni
step_sec = 10.0
results = propagator.propagate_at_fixed_times(duration_sec=duration_sec, step_sec=step_sec)

# Estrai risultati
r = [entry['position'] for entry in results]
v = [entry['velocity'] for entry in results]
A = [entry['a'] for entry in results]
I = [np.degrees(entry['i']) for entry in results]
time_list = [entry['date'].durationFrom(epoch) for entry in results]

# Calcola densità atmosferica
densities = []
for entry in results:
    date = entry['date']
    position = entry['position']
    density = atmosphere.getDensity(date, position, propagator.inertial_frame)
    densities.append(density)

# Salva dati in file
data = np.array([[ri.getX(), ri.getY(), ri.getZ(),
                  vi.getX(), vi.getY(), vi.getZ(), d] for ri, vi, d in zip(r, v, densities)])
np.savetxt("state_vectors_with_density.txt", data,
           header="X[m] Y[m] Z[m] VX[m/s] VY[m/s] VZ[m/s] Density[kg/m^3]",
           fmt="%.6e")

# Plot orbita
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

# Plot inclinazione
plt.figure()
plt.plot(time_list, I)
plt.xlabel("Time [s]")
plt.ylabel("Inclination [deg]")
plt.title("Inclination over time")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot semiasse
plt.figure()
plt.plot(time_list, A)
plt.xlabel("Time [s]")
plt.ylabel("Semi-major axis")
plt.title("Semi-major axis over time")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot densità
plt.figure()
plt.plot(time_list, densities)
plt.xlabel("Time [s]")
plt.ylabel("Density [kg/m^3]")
plt.title("Atmospheric density over time")
plt.yscale("log")  # scala logaritmica per valori piccoli
plt.grid(True)
plt.tight_layout()
plt.show()

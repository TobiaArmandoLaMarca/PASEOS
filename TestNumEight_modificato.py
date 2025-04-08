
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

sat_actor = ActorBuilder.get_actor_scaffold(name="Kanyini",actor_type=SpacecraftActor,epoch= pk.epoch_from_string("2025-03-05 19:00:28"))
earth = pk.planet.jpl_lp("earth")

ActorBuilder.set_orbit(actor=sat_actor,
    position=[-5529688.37, 4081251.929, 1308.67035],
    velocity=[593.22742, 783.20028099, 7553.094223426],
    epoch= pk.epoch_from_string("2025-03-05 19:00:28"),  # Sincronizza con l'epoca iniziale
    central_body=earth
)

t0 = pk.epoch_from_string("2025-03-05 19:00:28")  # Tempo iniziale
# Initialize the orekit virtual machine and set the current directory
vm = orekit.initVM()
setup_orekit_curdir()

# Set a start date
utc = TimeScalesFactory.getUTC()
epoch = AbsoluteDate(2025, 3, 5, 19, 00, 28.000, utc)
cfg = paseos.load_default_cfg()
#cfg.sim.time_multiplier = 25.0  # Acceleriamo la simulazione di 100x
cfg.sim.start_time = t0.mjd2000 * 86400  # Convertiamo MJD2000 in secondi
paseos_instance = paseos.init_sim(sat_actor, cfg)  # Avvia la simulazione

sat_attitude = Attitude(sat_actor)
eo_tools = EOTools(
    local_actor=sat_actor,
    actor_initial_attitude_in_deg=[0.0, 0.0, 0.0],
    actor_FOV_ACT_in_deg=[20.0],
    actor_FOV_ALT_in_deg=[20.0],
    actor_pointing_vector_body=[0.0, 0.0, 1.0]
)

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
numerical = propagator.propagator_num
state = numerical.getInitialState()

# Parametri target e assetto
target_geodetic = (math.radians(30.0), math.radians(45.0), 0.0)
eul_ang = [0.0, 0.0, 0.0]
z_brf = np.array([[0], [0], [1]])

# Parametri temporali
duration_sec = 184400.0
step_sec = 10.0
t_elapsed = 0.0
phi_rad = 0
# Funzione ausiliaria
def Point_Geodetic2ECI(lat, lon, alt, time):
    x, y, z = pm.geodetic2ecef(lat, lon, alt)
    T2 = pm.ecef2eci(x, y, z, time)
    x, y, z = T2
    return np.array([[x], [y], [z]])


# File output
output_file = "propagation_output_with_angle.txt"
with open(output_file, "w") as f:
    f.write("date,x,y,z,vx,vy,vz,angle_deg,insight,yaw_deg,pitch_deg,roll_deg\n")

    while t_elapsed <= duration_sec:
        new_date = state.getDate().shiftedBy(step_sec)
        new_state = numerical.propagate(new_date)
        pv = new_state.getPVCoordinates()
        pos = pv.getPosition()
        vel = pv.getVelocity()
        date_str = absolutedate_to_datetime(new_date).isoformat()

        x, y, z = pos.getX(), pos.getY(), pos.getZ()
        vx, vy, vz = vel.getX(), vel.getY(), vel.getZ()

        # Calcolo angolo con target
        angle_rad, vec_brf = eo_tools.off_nadir_pointing_angle(
                   z_brf=z_brf,
                   r_eci=np.array([[x], [y], [z]]),
                   v_eci=np.array([[vx], [vy], [vz]]),
                   target_geodetic=target_geodetic,
                   eul_angles_deg=eul_ang,
                   time=absolutedate_to_datetime(new_date)
                   )
        angle_deg = np.degrees(angle_rad)
        Insight = eo_tools.is_in_sight(target_geodetic = target_geodetic, 
                                       r_eci=np.array([[x], [y], [z]]), 
                                       v_eci=np.array([[vx], [vy], [vz]]), 
                                       time=absolutedate_to_datetime(new_date))
        yaw, pitch, roll = eo_tools.pointing_attitude(z_brf,vec_brf,phi_rad,eul_ang,Insight)

        #print(f"[{date_str}] θ = {angle_deg:.2f}°")
        f.write(
          f"{date_str},"
          f"{x:.3f},{y:.3f},{z:.3f},"
          f"{vx:.6f},{vy:.6f},{vz:.6f},"
          f"{angle_deg:.4f},{Insight},"
          f"{yaw:.2f},{pitch:.2f},{roll:.2f}\n"
        )

        state = new_state
        numerical.setInitialState(state)
        t_elapsed += step_sec

#print(f"\n✅ Dati salvati in {os.path.abspath(output_file)}")

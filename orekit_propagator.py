from org.orekit.orbits import KeplerianOrbit, PositionAngleType
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import Constants
from org.orekit.frames import FramesFactory
from org.orekit.orbits import OrbitType
from org.orekit.propagation.numerical import NumericalPropagator
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.sampling import PythonOrekitFixedStepHandler
from org.orekit.propagation.sampling import OrekitStepNormalizer
from org.orekit.orbits import OrbitType
from org.orekit.utils import IERSConventions
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity import ThirdBodyAttraction
from org.orekit.forces.radiation import SolarRadiationPressure
from org.orekit.forces.radiation import IsotropicRadiationSingleCoefficient
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid 
from org.orekit.models.earth.atmosphere import NRLMSISE00
from org.orekit.models.earth.atmosphere.data import CssiSpaceWeatherData
from org.orekit.forces.drag import DragForce, IsotropicDrag
from org.orekit.data import DataContext
from org.orekit.data import DataProvidersManager, DirectoryCrawler
from java.io import File
from orekit import JArray_double

import numpy as np


class OrekitPropagator:
    minStep = 1e-6
    maxstep = 1000.0
    initStep = 60.0
    positionTolerance = 1e-3

    def __init__(self, orbital_elements: list, epoch: AbsoluteDate, satellite_mass: float, area_s: float, cr_s: float, area_d: float, cd: float) -> None:
        inertialFrame = FramesFactory.getEME2000()
        a, e, i, omega, raan, lv = orbital_elements
        self.initialDate = epoch

        initialOrbit = KeplerianOrbit(
            a, e, i, omega, raan, lv,
            PositionAngleType.TRUE,
            inertialFrame,
            epoch,
            Constants.WGS84_EARTH_MU
        )

        tolerances = NumericalPropagator.tolerances(
            self.positionTolerance, initialOrbit, initialOrbit.getType()
        )

        integrator = DormandPrince853Integrator(
            self.minStep,
            self.maxstep,
            JArray_double.cast_(tolerances[0]),
            JArray_double.cast_(tolerances[1]),
        )
        integrator.setInitialStepSize(self.initStep)

        initialState = SpacecraftState(initialOrbit, satellite_mass)

        self.propagator_num = NumericalPropagator(integrator)
        self.propagator_num.setOrbitType(OrbitType.CARTESIAN)
        self.propagator_num.setInitialState(initialState)

        gravityProvider = GravityFieldFactory.getNormalizedProvider(70, 70)
        self.propagator_num.addForceModel(
            HolmesFeatherstoneAttractionModel(
                FramesFactory.getITRF(IERSConventions.IERS_2010, True), gravityProvider
            )
        )

        moon = CelestialBodyFactory.getMoon()
        self.propagator_num.addForceModel(ThirdBodyAttraction(moon))

        sun = CelestialBodyFactory.getSun()
        self.propagator_num.addForceModel(ThirdBodyAttraction(sun))

        wgs84_ellipsoid = OneAxisEllipsoid(
            Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
            Constants.WGS84_EARTH_FLATTENING,
            FramesFactory.getITRF(IERSConventions.IERS_2010, True),
        )

        radiation_model = IsotropicRadiationSingleCoefficient(area_s, cr_s)
        srp_model = SolarRadiationPressure(sun, wgs84_ellipsoid, radiation_model)
        self.propagator_num.addForceModel(srp_model)

        # Set up space weather and atmosphere
        orekitData = File("C:/Users/LaMar/miniforge3/envs/esaenv/Lib/site-packages/paseos/orekit-data/CSSI-Space-Weather-Data")
        manager = DataContext.getDefault().getDataProvidersManager()
        manager.addProvider(DirectoryCrawler(orekitData))
        utc = TimeScalesFactory.getUTC()
        cssi = CssiSpaceWeatherData(CssiSpaceWeatherData.DEFAULT_SUPPORTED_NAMES, manager, utc)
        atmosphere = NRLMSISE00(cssi, sun, wgs84_ellipsoid).withSwitch(9, -1)
        self.atmosphere = atmosphere
        #import pdb; pdb.set_trace()
        drag_model = DragForce(atmosphere, IsotropicDrag(area_d, cd))
        self.propagator_num.addForceModel(drag_model)

    def eph(self, time_since_epoch_in_seconds: float):
        state = self.propagator_num.propagate(
            self.initialDate, self.initialDate.shiftedBy(time_since_epoch_in_seconds)
        )
        return state

    def propagate_at_fixed_times(self, duration_sec: float, step_sec: float = 10):
        t_array = [
            self.initialDate.shiftedBy(float(dt))
            for dt in np.arange(0, duration_sec, step_sec)
        ]

        results = []
        for date in t_array:
            state = self.propagator_num.propagate(date)
            orbit = KeplerianOrbit(state.getOrbit())
            pv = state.getPVCoordinates()
            pos = pv.getPosition()
            dens = self.atmosphere.getDensity(date, pos, state.getFrame())

            results.append({
                'date': state.getDate(),
                'position': pos,
                'velocity': pv.getVelocity(),
                'a': orbit.getA(),
                'e': orbit.getE(),
                'i': orbit.getI(),
                'raan': orbit.getRightAscensionOfAscendingNode(),
                'pa': orbit.getPerigeeArgument(),
                'ta': orbit.getTrueAnomaly(),
                'density': dens  # 👈 Densità salvata
            })
        return results

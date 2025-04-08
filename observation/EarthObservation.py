#New Class to determine Observation Area of Interest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from astropy.time import Time
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import contextily as ctx
import time
import pdb 
import pandas as pd
import pymap3d as pm
import xml.etree.ElementTree as ET
from shapely.geometry import Point, Polygon
from loguru import logger
from paseos.actors.spacecraft_actor import SpacecraftActor
from shapely.geometry import Polygon, Point
from shapely.ops import transform
import pyproj
from pyproj import Geod
import warnings
from paseos.utils.constants import R_p, R_e
from datetime import datetime
from PIL import Image

from paseos.utils.reference_frame_transformation import(
    LVLH2IRF,
    RotMat_by_quat,
    IRF2LVLH,
    LVLH2BRF_eul,
    BRF2LVLH_eul,
    BRF2IRF_eul,
    IRF2BRF_eul,
    RotMat_LVLH_to_BRF_by_eul,
    rotation_matrix_to_ypr
)

from ..utils.point_trasformation import(
    Point_ECI2Geodetic, Point_Geodetic2ECI
)

class EOTools:
    #Spacecraft_actor.
    _actor = None
    #Actor attitude in deg
    _actor_attitude_in_deg = None
    #Actor pointing vector expressed in inertial frame.
    _actor_pointing_vector_eci = None
    #Actor pointing vector expressed in body reference frame
    _actor_pointing_vector_body = None
    #Earth Polar Radius


    """
    This class is provided with all the functions needed to perform the dedicated Earth-Observation activities
    """

    def __init__(
        self,
        local_actor,
        actor_initial_attitude_in_deg: list[float] = [0.0, 0.0, 0.0],
        actor_FOV_ACT_in_deg : list[float] = [1.0],
        actor_FOV_ALT_in_deg: list[float] = [1.0],
        actor_pointing_vector_body: list[float] = [0.0, 0.0, 1.0],
    ):

        assert isinstance(local_actor, SpacecraftActor), (
           "local_actor must be a " "SpacecraftActor" "."
        )
     
        logger.trace("Initializing EOTools Function")
        self._actor = local_actor
        
        # Convert attitude in np.ndarray
        self._actor_attitude_in_deg = np.array(actor_initial_attitude_in_deg)
        # Convert pointing vector 
        self._actor_pointing_vector_body = np.array(actor_pointing_vector_body)/np.linalg.norm(np.array(actor_pointing_vector_body))
        
        # Creation of the FOV
        self.fov_angles = [actor_FOV_ACT_in_deg[0], actor_FOV_ALT_in_deg[0]]


        # Creation of the rectangular FOV
    def get_fov_vectors_in_BRF(self):
        theta_x = np.deg2rad(self.fov_angles[0])
        theta_y = np.deg2rad(self.fov_angles[1])
        V1 = [-np.tan(theta_x/2), -np.tan(theta_y/2), 1]  # Top-right
        d1 = V1/np.linalg.norm(V1)
        V2 = [-np.tan(theta_x/2),  np.tan(theta_y/2), 1]  # Top-left
        d2 = V2/np.linalg.norm(V2)
        V3 = [ np.tan(theta_x/2),  np.tan(theta_y/2), 1]  # Bottom-left
        d3 = V3/np.linalg.norm(V3)
        V4 = [ np.tan(theta_x/2), -np.tan(theta_y/2), 1]  # Bottom-right
        d4 = V4/np.linalg.norm(V4)

        return np.array([d1, d2, d3, d4])

    def _find_intersection_in_Geodetic(self, ray_direction, eul_ang,time,r,v):

        Tullio = self._actor.local_time
        #import pdb; pdb.set_trace()
        #r = np.array(self._actor.get_position(self._actor.local_time))
        x_ECI, y_ECI, z_ECI = r[0], r[1], r[2]
        #v = np.array(self._actor.get_position_velocity(self._actor.local_time)[1])
        #x_ECEF, y_ECEF, z_ECEF = pm.eci2ecef(x_ECI, y_ECI, z_ECI,self._actor.local_time) 
        #Rays direction in ECEF
        d_in_ECI = []
        for ray in ray_direction:
            new_vector = BRF2IRF_eul(ray,r,v,eul_ang)  
            new_vector = new_vector
            #import pdb; pdb.set_trace()
            d_in_ECI.append(new_vector)
        #import pdb; pdb.set_trace()

        intersections = []
        for d in d_in_ECI:
            dx, dy, dz = d 
            # Coefficienti dell'equazione quadratica
            A = (dx**2 / R_e**2) + (dy**2 / R_e**2) + (dz**2 / R_p**2)
            B = 2 * ((x_ECI * dx / R_e**2) + (y_ECI * dy / R_e**2) + (z_ECI * dz / R_p**2))
            C = (x_ECI**2 / R_e**2) + (y_ECI**2 / R_e**2) + (z_ECI**2 / R_p**2) - 1

            # Calcolo del discriminante
            delta = B**2 - 4*A*C
            #import pdb; pdb.set_trace()
            if delta < 0:
                continue  # Nessuna intersezione
            # Soluzioni per t
            t1 = (-B + np.sqrt(delta)) / (2*A)
            t2 = (-B - np.sqrt(delta)) / (2*A)

            # Scegliamo il valore minimo positivo di t
            if min(t1, t2) > 0:
                t = min(t1, t2)
            else:
                t = max(t1, t2)
        
            if t < 0:
                continue
            
    # print("Punti di intersezione non sufficienti")
    # return None
                # Punto di intersezione
            intersection_ECI = r + t * d
            intersection_Geod = Point_ECI2Geodetic(intersection_ECI[0], intersection_ECI[1], intersection_ECI[2], time)
            intersections.append(intersection_Geod)
        
        if len(intersections) < 4:
            raise ValueError("Punti di intersezione non sufficienti")
        
    # Oppure, se preferisci stampare un messaggio:
        intersections_matrix = np.column_stack(intersections)
        
        return intersections_matrix
    
    def load_kml(self,file_path):
    
    
        tree = ET.parse(file_path)
        root = tree.getroot()
    
        # Namespace KML
        ns = {'kml': 'http://www.opengis.net/kml/2.2', 'ns0': 'http://www.opengis.net/kml/2.2'}
    
        placemarks = []
        for placemark in root.findall(".//kml:Placemark", ns):
            name = placemark.find("kml:name", ns).text if placemark.find("kml:name", ns) is not None else None
        
            # Estrarre TimeSpan Begin e End
            timespan = placemark.find(".//kml:TimeSpan", ns)
            begin_time = timespan.find("kml:begin", ns).text if timespan is not None and timespan.find("kml:begin", ns) is not None else None
            end_time = timespan.find("kml:end", ns).text if timespan is not None and timespan.find("kml:end", ns) is not None else None
        
            polygon = placemark.find(".//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", ns)
        
            if polygon is not None:
               coords_text = polygon.text.strip()
               coords_list = [tuple(map(float, coord.split(","))) for coord in coords_text.split()]
            else:
               coords_list = None

            # Aggiungere al dataset
            placemarks.append({
               "name": name,
               "begin_time": begin_time,
               "end_time": end_time,
               "polygon": coords_list
            })
    
        # Convertire in DataFrame
        df = pd.DataFrame(placemarks)
    
        # Convertire i timestamp in formato datetime
        df['begin_time'] = pd.to_datetime(df['begin_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])
    
        return df
    
    def off_nadir_pointing_angle(self,z_brf, r_eci, v_eci, target_geodetic, eul_angles_deg, time):
         # Step 1: target geodetico → ECI
        P_target_eci = Point_Geodetic2ECI(*target_geodetic, time)

         # Step 2: vettore dal satellite al target in ECI
        vec_eci = P_target_eci - r_eci
        #import pdb; pdb.set_trace()
        # Step 3: rotazione ECI → BRF
        vec_brf = IRF2BRF_eul(vec_eci,r_eci, v_eci, eul_angles_deg)

        # Step 4: normalizza e calcola l’angolo tra z_brf e il vettore target
        z_brf = z_brf / np.linalg.norm(z_brf)
        vec_brf = vec_brf / np.linalg.norm(vec_brf)
        dot_product = np.clip(z_brf.T @ vec_brf, -1.0, 1.0)
        angle_rad = np.arccos(dot_product)

        return float(angle_rad), vec_brf
    
    def is_in_sight(self,target_geodetic,r_eci,v_eci,time):
        lat, lon, alt = target_geodetic
        Ecef_satellite = pm.eci2ecef(r_eci[0],r_eci[1],r_eci[2], time)
        e, n, u = pm.ecef2enu(Ecef_satellite[0], Ecef_satellite[1], Ecef_satellite[2], target_geodetic[0], target_geodetic[1], target_geodetic[2])
        az, el, _ = pm.enu2aer(e, n, u)
        in_sight = el > 0
        return in_sight
    
    def pointing_attitude(self,l1,l2,phi_rad,attitude,is_in_view):

        if not is_in_view:
           return np.nan, np.nan, np.nan
        
        l1 = np.asarray(l1).flatten()
        l2 = np.asarray(l2).flatten()
        l1 = l1 / np.linalg.norm(l1)
        l2 = l2 / np.linalg.norm(l2)
        dot = np.clip(np.dot(l1, l2), -1.0, 1.0)
        cross = np.cross(l1, l2)
        cos_phi_2 = np.cos(phi_rad / 2)
        sin_phi_2 = np.sin(phi_rad / 2)
        numerator_vec = cross * cos_phi_2 + (l1 + l2) * sin_phi_2
        numerator_scalar = (1 + dot) * cos_phi_2
        denominator = np.sqrt(2 * (1 + dot))
        q_vec = numerator_vec / denominator
        q_scalar = numerator_scalar / denominator
        vec_fin = np.concatenate((q_vec, [q_scalar]))
        Rot_SRFa_SRFp = RotMat_by_quat(vec_fin)
        Rot_LVLH2SRFp =  Rot_SRFa_SRFp @ RotMat_LVLH_to_BRF_by_eul(attitude)
        yaw, pitch, roll = rotation_matrix_to_ypr(Rot_LVLH2SRFp)
        yaw = np.degrees(yaw)
        pitch = np.degrees(pitch)
        roll = np.degrees(roll)
        return yaw, pitch, roll



    
    @staticmethod
    def plot_fov_on_map(intersections, ax):
        
         #ax.clear()
         Image.MAX_IMAGE_PIXELS = None
         #ax.stock_img()
         #ax.add_feature(cfeature.BORDERS, linestyle = ':')
         #ax.add_feature(cfeature.COASTLINE)
         # Griglia
         #ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, color='gray')
             # Estrai le coordinate latitudine e longitudine dei punti FOV
         
         #tiler = cimgt.OSM()
         #ax.add_image(tiler, 4)  # Zoom livello 4 (puoi aumentarlo per più dettagli)
         
         #ax.set_global()
         
         #ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=3)
         #ax.set_global()
         

         img = mpimg.imread(r"C:\Users\LaMar\miniforge3\envs\esaenv\Lib\site-packages\paseos\WorldMap2.jpg")
         # Disegna griglia e confini
         ax.imshow(img, extent=[-180, 180, -90, 90], transform=ccrs.PlateCarree())
         ax.add_feature(cfeature.BORDERS, linestyle=":")
         ax.add_feature(cfeature.COASTLINE)
         ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, color='gray')
         ax.set_global()

         latitudes = intersections[:, 0]
         longitudes = intersections[:, 1]

         # Chiudiamo il riquadro del FOV collegando il primo e l'ultimo punto
         latitudes = np.append(latitudes, latitudes[0])
         longitudes = np.append(longitudes, longitudes[0])

         # Plottiamo il FOV sulla mappa
         ax.plot(longitudes, latitudes, color='red', linestyle='-', transform=ccrs.PlateCarree())
         # Disegniamo l'area del FOV con riempimento rosso trasparente
         ax.fill(longitudes, latitudes, color='red', alpha=0.1, transform=ccrs.PlateCarree())  # alpha=0.3 rende il colore trasparente

         # Disegniamo i punti di intersezione con marker più piccoli
         ax.scatter(longitudes, latitudes, color='black', s=0.001, transform=ccrs.PlateCarree())  # s=10 riduce la dimensione dei punti
         #plt.title("Field of View (FOV) sulla Mappa Geografica")
         plt.pause(0.0005)  # Attende un attimo per la visualizzazione

    @staticmethod   
    def check_fov_in_polygon(df, simulation_time, fov_vertices):
        fov_vertices = [(lon, lat) for lat, lon in fov_vertices]
        # Assicurarsi che i tempi siano in formato datetime
        df['begin_time'] = pd.to_datetime(df['begin_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])

        # Trovare il poligono con il begin_time più vicino alla simulation_time
        df['time_diff'] = abs(df['begin_time'] - simulation_time)
        closest_row = df.loc[df['time_diff'].idxmin()]

        # Estrarre il poligono corrispondente
        polygon_coords = closest_row['polygon']
        polygon = Polygon(polygon_coords) if polygon_coords else None
        geod = Geod(ellps="WGS84")
        fov_inside = False
        coverage_ratio = 0
        inside_count = 0
        poly_area = 0
        fov_area = 0
        intersection_area_km2 = 0  # 👈 Safe default
        area_km2 = 0

        if polygon:
           inside_count = sum(Point(pt).within(polygon) for pt in fov_vertices)
           if inside_count >=2 :
              # Creare il poligono del FOV
              fov_polygon = Polygon(fov_vertices)
              project = pyproj.Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True).transform              
              with warnings.catch_warnings():
                   warnings.simplefilter("error", RuntimeWarning) 
                   try:
                        intersection_polygon = fov_polygon.intersection(polygon)
                        if not intersection_polygon.is_empty:
                          # polygon.exterior.coords dà una lista di (lon, lat)
                            coords_fov = list(fov_polygon.exterior.coords)
                            lon_fov = [pt[0] for pt in coords_fov]
                            lat_fov = [pt[1] for pt in coords_fov]
                            area_fov = abs(geod.polygon_area_perimeter(lon_fov, lat_fov)[0])/1e6
                            coords_int = list(intersection_polygon.exterior.coords)
                            lon_int = [pt[0] for pt in coords_int]
                            lat_int = [pt[1] for pt in coords_int]
                            area_int = abs(geod.polygon_area_perimeter(lon_int, lat_int)[0])/1e6
                            coverage_ratio = (area_int / area_fov) * 100 if area_fov > 0 else 0
                            intersection_area_km2 = area_int
                            #pdb.set_trace() 
                        
                        fov_inside = inside_count >= 2 and coverage_ratio > 90
                        
                   except RuntimeWarning:
                        print("⚠️ Intersezione fallita, avviando debug mode...")
                        pdb.set_trace()  # Ti permette di ispezionare variabili quando il problema si verifica   
        return {
            'selected_polygon': polygon,
            'fov_inside': fov_inside,
            'coverage_ratio': coverage_ratio if polygon else 0,
            'inside_count': inside_count if polygon else 0,
            'intersection_area_km2' : intersection_area_km2 if polygon else 0,
            'closest_time': closest_row['begin_time'],
            'area_id': closest_row['name']
        }
    
    @staticmethod

    def vec3d_to_list(v):
        return np.array([v.getX(), v.getY(), v.getZ()])

         
def plot_fov_on_map_vid(intersections, ax):
        
         #ax.clear()
         Image.MAX_IMAGE_PIXELS = None
         #ax.stock_img()
         #ax.add_feature(cfeature.BORDERS, linestyle = ':')
         #ax.add_feature(cfeature.COASTLINE)
         # Griglia
         #ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, color='gray')
             # Estrai le coordinate latitudine e longitudine dei punti FOV
         
         #tiler = cimgt.OSM()
         #ax.add_image(tiler, 4)  # Zoom livello 4 (puoi aumentarlo per più dettagli)
         
         #ax.set_global()
         
         #ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=3)
         #ax.set_global()
         

         img = mpimg.imread(r"C:\Users\LaMar\miniforge3\envs\esaenv\Lib\site-packages\paseos\WorldMap2.jpg")
         # Disegna griglia e confini
         ax.imshow(img, extent=[-180, 180, -90, 90], transform=ccrs.PlateCarree())
         ax.add_feature(cfeature.BORDERS, linestyle=":")
         ax.add_feature(cfeature.COASTLINE)
         ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, color='gray')
         ax.set_global()

         latitudes = intersections[:, 0]
         longitudes = intersections[:, 1]

         # Chiudiamo il riquadro del FOV collegando il primo e l'ultimo punto
         latitudes = np.append(latitudes, latitudes[0])
         longitudes = np.append(longitudes, longitudes[0])

         # Plottiamo il FOV sulla mappa
         ax.plot(longitudes, latitudes, color='red', linestyle='-', transform=ccrs.PlateCarree())
         # Disegniamo l'area del FOV con riempimento rosso trasparente
         ax.fill(longitudes, latitudes, color='red', alpha=0.1, transform=ccrs.PlateCarree())  # alpha=0.3 rende il colore trasparente

         # Disegniamo i punti di intersezione con marker più piccoli
         ax.scatter(longitudes, latitudes, color='black', s=0.001, transform=ccrs.PlateCarree())  # s=10 riduce la dimensione dei punti
         #plt.title("Field of View (FOV) sulla Mappa Geografica")
         #plt.pause(0.0005)  # Attende un attimo per la visualizzazione







         
        

         


        

        


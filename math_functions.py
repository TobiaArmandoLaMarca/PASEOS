import numpy as np
import matplotlib.pyplot as plt


def latlon_to_xyz(lat, lon, alt, R_earth):

    lat = np.radians(lat)
    lon = np.radians(lon)

    x = (R_earth + alt) * np.cos(lat) * np.cos(lon)
    y = (R_earth + alt) * np.cos(lat) * np.sin(lon)
    z = (R_earth + alt) * np.sin(lat)

    return np.array([x, y, z]).T

def compute_nadir_angle(sat_lat, sat_lon, sat_alt, feature_lat, feature_lon, R_earth):

    # xyz coordinates
    sat_position = latlon_to_xyz(sat_lat, sat_lon, sat_alt, R_earth).T
    feature_position = latlon_to_xyz(feature_lat, feature_lon, 0, R_earth).T

    # Vectors
    sat_to_feature = feature_position - sat_position
    sat_to_nadir = - sat_position  # wrt np.zeros(3), as the Earth center is at (0,0,0)

    # Angle
    theta_rad = np.arccos(np.dot(sat_to_feature, sat_to_nadir) / (np.linalg.norm(sat_to_feature) * np.linalg.norm(sat_to_nadir)))
    nadir_angle_deg = np.degrees(theta_rad)

    if np.isnan(nadir_angle_deg):
        nadir_angle_deg = 0

    return nadir_angle_deg


def check_line_of_sight(satellite, feature, R_earth):

    # Satellite and feature positions
    x_s, y_s, z_s = satellite
    x_f, y_f, z_f = feature

    # Direction vector from satellite to feature
    dx = x_f - x_s
    dy = y_f - y_s
    dz = z_f - z_s

    # Calculate coefficients of the quadratic equation
    A = dx ** 2 + dy ** 2 + dz ** 2
    B = 2 * (dx * x_s + dy * y_s + dz * z_s)
    C = x_s ** 2 + y_s ** 2 + z_s ** 2 - R_earth ** 2

    # Discriminant of the quadratic equation
    discriminant = B ** 2 - 4 * A * C

    # If the discriminant is negative, there's no intersection
    if discriminant < 0:
        print("Feature is in line of sight")
        in_sight = True  # No intersection with the Earth (feature is visible)

    else:

        # Calculate the two possible t values
        t1 = (-B - np.sqrt(discriminant)) / (2 * A)
        t2 = (-B + np.sqrt(discriminant)) / (2 * A)

        if 0 <= t1 <= 1 and 0 <= t2 <= 1:
            print("Feature is out of sight")
            # Two surface crossings, so out of line of sight
            in_sight =  False

        else:
            print("Feature is in line of sight")
            in_sight = True



    return in_sight
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin

import matplotlib.pyplot as plt
from math_functions import latlon_to_xyz, compute_nadir_angle, check_line_of_sight
from plot_functions import plot_earth_and_line, plot_earth_slice, plot_result, plot_earth_with_pyvista

# Input parameters
sat_lat = 52                     # Latitude [deg]
sat_lon = 4                      # Longitude [deg]
sat_alt = 786 * 10**3            # Satellite altitude [m]

focal_length = 0.440             # Focal length of Sentinel [m]
R_earth = 6378137.0              # Radius Earth [m]
highest_point = 512              # Highest point in input image [m] - quick fix for not working DEM input

# Input paths
input_image  = "C:/Users/LaMar/miniforge3/envs/esaenv/Lib/site-packages/paseos/test_data/B01_2.tiff"                # GeoTIFF file
input_dem    = "C:/Users/LaMar/miniforge3/envs/esaenv/Lib/site-packages/paseos/test_data/DEM_2.tiff"  # or None     # DEM file
output_image = "C:/Users/LaMar/miniforge3/envs/esaenv/Lib/site-packages/paseos/test_data/test_output.tiff"

plot_line_of_sight = True

with rasterio.open(input_image) as src_input:

    if input_dem == None:
        data_dem_norm = np.zeros((src_input.height, src_input.width), dtype=np.float32)

    else:
        with rasterio.open(input_dem) as src_dem:

            # Align DEM file with input tiff file
            data_dem = np.empty((src_input.height, src_input.width), dtype=np.float32)

            reproject(
                source=rasterio.band(src_dem, 1),
                destination=data_dem,
                src_transform=src_dem.transform,
                src_crs=src_dem.crs,
                dst_transform=src_input.transform,
                dst_crs=src_input.crs,
                resampling=Resampling.bilinear,
            )

            # Convert DEM data (quick fix)
            data_dem_norm = (data_dem - np.min(data_dem)) / (np.max(data_dem) - np.min(data_dem))
            data_dem_norm = data_dem_norm * highest_point



    # Extract data from input image
    bounds = src_input.bounds                               # Latitude and longitude bounds
    width, height = src_input.width, src_input.height
    data_input = src_input.read(1)                                # First band of the image
    transform_matrix = src_input.transform

    data_input_norm = (data_input - np.min(data_input)) / (np.max(data_input) - np.min(data_input)) * 255

    height, width = data_input_norm.shape
    print("Loaded image with size ", height, 'x', width)

    # Create array with latitude and longitudes for input image
    lon_min, lat_max, lon_max, lat_min = bounds.left, bounds.top, bounds.right, bounds.bottom
    lons = np.linspace(lon_min, lon_max, width)
    lats = np.linspace(lat_max, lat_min, height)
    img_lon, img_lat = np.meshgrid(lons, lats)

    img_lat_avg = np.mean(img_lat)
    img_lon_avg = np.mean(img_lon)
    print("Taken at ", img_lat_avg, 'N , ', img_lon_avg, 'E')
    print("Observed at ", sat_lat, 'N , ', sat_lon, 'E , ', np.round(sat_alt/1000,1), ' km altitude')

    # Compute nadir angle
    nadir_angle_deg = compute_nadir_angle(sat_lat, sat_lon, sat_alt, img_lat_avg, img_lon_avg, R_earth)
    print("With off-nadir angle ", np.round(nadir_angle_deg,2) , ' deg')

    # Convert lat lon of image and satellite to xyz (in earth centered frame)
    img_lat_flat = img_lat.ravel()
    img_lon_flat = img_lon.ravel()
    data_dem_flat = data_dem_norm.ravel()

    ecef_img = latlon_to_xyz(img_lat_flat, img_lon_flat, data_dem_flat, R_earth)
    ecef_sat = latlon_to_xyz(sat_lat, sat_lon, sat_alt, R_earth)
    ecef_northpole = np.array([0, 0, R_earth])  # y points to the south pole (easier for image plotting)

    ecef_img_avg = np.mean(ecef_img, axis=0)
    in_sight = check_line_of_sight(ecef_sat, ecef_img_avg, R_earth)

    if plot_line_of_sight:

        plot_earth_with_pyvista(ecef_sat, ecef_img_avg, R_earth=R_earth)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.axis('off')

        plot_earth_slice(ecef_sat, ecef_img_avg, R_earth=R_earth, ax2d=ax)
        plt.show()

        fig.savefig('line_of_sight.jpg')

       # # Create a figure with subplots (1 row, 2 columns)
       # fig = plt.figure(figsize=(18, 8))

#
       # # Create 3D plot for Earth with satellite and feature
       # ax3d = fig.add_subplot(121, projection='3d')
       # plot_earth_and_line(ecef_sat, ecef_img_avg, R_earth=R_earth, ax3d=ax3d)
#
       # # Create 2D plot for Earth slice projection
       # ax2d = fig.add_subplot(122)
       # plot_earth_slice(ecef_sat, ecef_img_avg, R_earth=R_earth, ax2d=ax2d)

       # # Show the plots
       # plt.tight_layout()
       # plt.show()

    if in_sight:
        # Define satellite reference frame
        e_z = -ecef_sat / np.linalg.norm(ecef_sat)                    # z points downwards to nadir
        north_vector = ecef_northpole - ecef_sat                      # x points North
        e_x = north_vector / np.linalg.norm(north_vector)
        e_y = np.cross(e_z, e_x)                                      # y completes the right-handed system

        # Transform the image from ecef to satellite-centered coordinates
        ecef_sat_to_img = ecef_img - ecef_sat
        crss_img = np.dot(ecef_sat_to_img, np.array([e_x, e_y, e_z]).T)         # crss is coordinate reference system satellite

        # Convert 3D coordinates to image coordinates with pinhole model
        x_projected = (crss_img[:, 1] / crss_img[:, 2]) * focal_length
        y_projected = -(crss_img[:, 0] / crss_img[:, 2]) * focal_length

        # Make projected coordinates in image
        x_norm = ((x_projected - x_projected.min()) / (x_projected.max() - x_projected.min()) * width).astype(int)
        y_norm = ((y_projected - y_projected.min()) / (y_projected.max() - y_projected.min()) * height).astype(int)

        # Create the output image
        output_image_array = np.zeros((height, width))


        for i, (x, y) in enumerate(zip(x_norm, y_norm)):
            if 0 <= x < width and 0 <= y < height:
                output_image_array[y, x] = data_input_norm[i // width, i % width]


        # Save the output
        transform = from_origin(bounds.left, bounds.top, transform_matrix[0], transform_matrix[4])  # Top-left corner
        metadata = src_input.meta
        metadata.update({
            'count': 1,  # One band in the output
            'dtype': 'uint8',  # Output image type
            'crs': src_input.crs,  # Same CRS as input
            'transform': transform  # Apply the original transform
        })

        with rasterio.open(output_image, 'w', **metadata) as dst:
            dst.write(output_image_array, 1)  # Write data to the first band
        print(f"Satellite view saved as {output_image}")

        src_out = rasterio.open(output_image)
        data_out = src_out.read(1)

        plot_result(data_input_norm, data_out)

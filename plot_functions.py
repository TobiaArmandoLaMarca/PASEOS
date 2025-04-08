import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvista
from pyvista import examples


# Function to plot the slice of Earth passing through the satellite and feature
def plot_earth_slice(satellite, feature, R_earth, ax2d):

    satellite_vector = np.array(satellite)
    feature_vector = np.array(feature)

    # Calculate the normal to the plane formed by the Earth center, satellite, and feature
    normal_vector = np.cross(satellite_vector, feature_vector)

    # Normalize the normal vector for the plane
    normal_vector = normal_vector / np.linalg.norm(normal_vector)


    # Plot Earth as a circle in 2D projection on the plane
    earth_circle = plt.Circle((0, 0), R_earth, color='lightblue', alpha=0.5, label='Earth')
    ax2d.add_patch(earth_circle)

    # Compute the rotation matrix to align the plane with the XY plane
    # The normal vector to the plane is already calculated, so we will align it with the z-axis (0, 0, 1)
    rotation_axis = np.cross(normal_vector, np.array([0, 0, 1]))  # Axis to rotate around (from normal to z-axis)
    rotation_angle = np.arccos(np.dot(normal_vector, np.array([0, 0, 1])))  # Angle to rotate by

    # Create the rotation matrix using the Rodrigues' rotation formula
    def rotation_matrix(axis, angle):
        axis = axis / np.linalg.norm(axis)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        ux, uy, uz = axis
        return np.array([
            [cos_angle + ux ** 2 * (1 - cos_angle), ux * uy * (1 - cos_angle) - uz * sin_angle,
             ux * uz * (1 - cos_angle) + uy * sin_angle],
            [uy * ux * (1 - cos_angle) + uz * sin_angle, cos_angle + uy ** 2 * (1 - cos_angle),
             uy * uz * (1 - cos_angle) - ux * sin_angle],
            [uz * ux * (1 - cos_angle) - uy * sin_angle, uz * uy * (1 - cos_angle) + ux * sin_angle,
             cos_angle + uz ** 2 * (1 - cos_angle)]
        ])

    # Apply the rotation to both the satellite and feature vectors
    rotation_mat = rotation_matrix(rotation_axis, rotation_angle)
    satellite_rotated = np.dot(rotation_mat, satellite_vector)
    feature_rotated = np.dot(rotation_mat, feature_vector)

    # Now we can plot the 2D projection on the slice defined by the satellite and feature
    x_s, y_s, _ = satellite_rotated  # Projected satellite coordinates in 2D
    x_f, y_f, _ = feature_rotated  # Projected feature coordinates in 2D

    # Draw the line of sight between the satellite and the feature
    ax2d.plot([x_s, x_f], [y_s, y_f], color='black', label='Line of Sight')

    # Plot the satellite and feature points in the 2D plane
    ax2d.scatter(x_s, y_s, color='teal', s=100, label='Satellite')
    ax2d.scatter(x_f, y_f, color='violet', s=100, label='Feature')


    # Set labels and title for 2D projection
    ax2d.set_xlabel('X')
    ax2d.set_ylabel('Y')
    ax2d.set_title('2D Projection of Earth Slice')

    # Set the aspect ratio to be equal (circle for Earth)
    ax2d.set_aspect('equal', 'box')

    distance = np.sqrt(x_s ** 2 + y_s ** 2)
    sat_alt = (distance - R_earth)

    margin = 1.5*sat_alt

    # Set the x and y limits with margin added around the Earth
    ax2d.set_xlim(-R_earth - margin, R_earth + margin)
    ax2d.set_ylim(-R_earth - margin, R_earth + margin)

    # Add a legend
    ax2d.legend()


# Function to plot the 3D Earth and line of sight
def plot_earth_and_line(satellite, feature, R_earth, ax3d):

    plt.close()

    # Generate sphere points
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = R_earth * np.outer(np.cos(u), np.sin(v))
    y = R_earth * np.outer(np.sin(u), np.sin(v))
    z = R_earth * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the sphere (Earth)
    # Set the axes to be equal
    ax3d.set_box_aspect([1, 1, 1])  # Aspect ratio is equal along all axes

    # Plot the Earth
    ax3d.plot_surface(x, y, z, color='lightblue', alpha=0.7, zorder = 0)

    # Plot the satellite and the feature points
    x_s, y_s, z_s = satellite
    x_f, y_f, z_f = feature

    ax3d.scatter(x_s, y_s, z_s, color='teal', s=100, label='Satellite', zorder=5)
    ax3d.scatter(x_f, y_f, z_f, color='violet', s=100, label='Feature', zorder=5)

    # Draw a line between the satellite and the feature
    ax3d.plot([x_s, x_f], [y_s, y_f], [z_s, z_f], color='black', label='Line of Sight')


    # Plot the three reference axes (X, Y, Z)
    ax3d.quiver(0, 0, 0, R_earth, 0, 0, color='grey', label='X Axis')  # X axis
    ax3d.quiver(0, 0, 0, 0, R_earth, 0, color='grey', label='Y Axis')  # Y axis
    ax3d.quiver(0, 0, 0, 0, 0, R_earth, color='grey', label='Z Axis')  # Z axis

    # Add labels to the end of the reference axes (X, Y, Z)
    ax3d.text(R_earth, 0, 0, 'X', color='grey', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
    ax3d.text(0, R_earth, 0, 'Y', color='grey', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
    ax3d.text(0, 0, R_earth, 'Z', color='grey', fontsize=12, verticalalignment='bottom', horizontalalignment='right')

    # Set labels and title
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_title('Earth with Satellite and Feature')
    # Show the legend
    ax3d.legend()

    # Set the view angle (rotate by 90 degrees around the z-axis)
    ax3d.view_init(elev=0, azim=0)  # Change azim to rotate the view

def plot_result(data1, data2):

    combined_data = np.array([data1, data2])
    # Get the min and max of all your data
    _min, _max = np.amin(combined_data[combined_data > 0]), np.amax(combined_data)

    x_shape = data1.shape[1]
    y_shape = data1.shape[0]

    plt.close()

    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(15)
    fig.set_figheight(6)

    plt.tight_layout(pad=0.05, w_pad=0.001, h_pad=2.0)
    ax1 = plt.subplot(121)  # creates first axis
    ax1.set_xticks(np.arange(0, x_shape, 200))
    ax1.set_yticks(np.arange(0, y_shape, 200))
    ax1.axis('equal')



    #ax1.grid('True')
    i1 = ax1.imshow(data1, cmap='pink')

    ax1.set_title("Input image", y=1.05, fontsize=12)
    ax2 = plt.subplot(122)  # creates second axis
    ax2.set_xticks(np.arange(0, x_shape, 200))
    ax2.set_yticks(np.arange(0, y_shape, 200))
    ax2.axis('equal')
    # ax2.axis('off')
    #ax2.grid('True')
    i1 = ax2.imshow(data2, cmap='pink')
    ax2.set_title("Output image", y=1.05, fontsize=12)


    plt.gcf().tight_layout()
    plt.show()

def plot_earth_with_pyvista(satellite, feature, R_earth):

    earth = examples.planets.load_earth(radius=R_earth)
    earth_texture = examples.load_globe_texture()

    satellite_copy = satellite.copy()
    feature_copy = feature.copy()

    pl = pyvista.Plotter(shape=(1, 1))

    pl.subplot(0, 0)
    pl.add_text("3D View", font_size = 12)
    pl.add_mesh(earth, texture=earth_texture, smooth_shading=True)
    pl.link_views()

    satellite_copy[0] = -satellite_copy[0]
    satellite_copy[1] = -satellite_copy[1]
    feature_copy[0] = -feature_copy[0]
    feature_copy[1] = -feature_copy[1]

    # Plot satellite (represented as a point)
    pl.add_points(satellite_copy, color="teal", point_size=16, render_points_as_spheres=True, label = 'Satellite')
    pl.add_points(feature_copy, color="violet", point_size=16, render_points_as_spheres=True, label = 'Feature')

    pl.add_lines(np.array([satellite_copy, feature_copy]), color='black', width=3)

    # Set view options
    pl.set_background('white')
    pl.view_isometric()
    pl.add_legend(bcolor='w', face='circle', size = (0.12, 0.12))

    pl.show(cpos="xy")
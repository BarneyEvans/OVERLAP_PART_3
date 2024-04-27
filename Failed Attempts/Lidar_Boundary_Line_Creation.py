from Plot_Fustum_PointCloud import unique_points, image_creation, seq_id, frame_id, frustum_dict
from Lidar_Boundary_Points import boundary_points, frustum_edges, pcd_lidar, strip_points
from scipy.spatial.distance import euclidean
import numpy as np
import time
import open3d as o3d
from scipy.spatial import ConvexHull, Delaunay
from scipy.interpolate import splprep, splev
import numpy.linalg as la

# Dictionary to hold line equations by strip
strip_lines = {}

for strip_name, points in strip_points.items():
    # Convert points list to numpy array
    points_array = np.array(points)

    # Fit a line to the points using least squares (or another method depending on shape)
    # This example assumes points are mostly linear
    if points_array.size > 0:
        A = np.c_[points_array[:, 0], np.ones(points_array.shape[0])]
        B = points_array[:, 1]
        coeff, _, _, _ = la.lstsq(A, B, rcond=None)
        strip_lines[strip_name] = coeff


# Define a function to generate line points
def generate_line_points(coeff, x_values):
    return coeff[0] * x_values + coeff[1]


# Dictionary to hold the line of coordinates for visualization
strip_line_points = {}

for strip_name, coeff in strip_lines.items():
    # Generate x values within the range of the strip
    x_values = np.linspace(min(strip_points[strip_name], key=lambda x: x[0])[0],
                           max(strip_points[strip_name], key=lambda x: x[0])[0], 100)
    y_values = generate_line_points(coeff, x_values)

    # Combine x and y to form coordinates
    line_points = np.column_stack((x_values, y_values, np.zeros_like(x_values)))  # Assuming Z=0 for simplicity
    strip_line_points[strip_name] = line_points

# Create Open3D point cloud objects for each strip line
for strip_name, line_points in strip_line_points.items():
    line_pcd = o3d.geometry.PointCloud()
    line_pcd.points = o3d.utility.Vector3dVector(line_points)
    line_pcd.paint_uniform_color([1, 0, 0])  # Red color for the strip lines
    o3d.visualization.draw_geometries([line_pcd], window_name=strip_name)

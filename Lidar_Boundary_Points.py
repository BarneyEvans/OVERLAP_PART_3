import os

from Plot_Fustum_PointCloud import unique_points, image_creation, seq_id, frame_id, frustum_dict
from scipy.spatial.distance import euclidean
import numpy as np
import time
import open3d as o3d
from scipy.spatial import ConvexHull, Delaunay
from collections import defaultdict
from Determining_Threshold_Points import is_near_boundary_and_within_edge, distance_to_line_along_local_y


def extract_top_edges(frustums):
    top_edges = {}
    for camera, points in frustums.items():
        top_edges[camera] = [
            (points[0], points[4]),
            (points[1], points[5])
        ]
    return top_edges

def define_line(p1, p2):
    """Create a parametric line function from two points."""
    direction = p2 - p1  # The direction vector from p1 to p2
    return lambda t: p1 + t * direction

def define_local_y_axis(p1, p2):
    """Define a local Y-axis that is orthogonal to the line segment in the XY plane."""
    global_z = np.array([0, 0, 1])  # Global Z-axis
    edge_vector = np.array(p2) - np.array(p1)
    edge_vector[2] = 0  # Ignore the Z component
    local_y = np.cross(global_z, edge_vector)  # Cross product to find a vector orthogonal in the XY plane
    local_y_normalized = local_y / np.linalg.norm(local_y)  # Normalize the vector
    return local_y_normalized

def distance_to_line_along_local_y(point, p1, p2):
    """Calculate the perpendicular distance from a point to a line along the local Y-axis."""
    local_y_axis = define_local_y_axis(p1, p2)
    point_vector = np.array(point) - np.array(p1)
    point_vector[2] = 0  # Ignore the Z component
    # Project the point vector onto the local Y-axis
    projection_length = np.dot(point_vector, local_y_axis)
    return abs(projection_length)

# Assuming 'frustum_dict' and 'unique_points' are defined elsewhere
frustum_edges = extract_top_edges(frustum_dict)
point_line_distances = {i: [] for i, _ in enumerate(unique_points)}

# Iterate over each point and each line
for i, point in enumerate(unique_points):
    for cam, edges in frustum_edges.items():
        for edge_idx, edge_points in enumerate(edges):
            p1, p2 = edge_points  # Unpack the two points defining the line

            # Calculate the distance from the current point to this line along the local Y-axis
            distance = distance_to_line_along_local_y(point, p1, p2)

            # Store the distance along with the camera, edge index, and local Y-axis
            point_line_distances[i].append({'camera': cam, 'edge_idx': edge_idx, 'distance': distance, 'edge_coordinates': edge_points, 'lidar_point': point})

# Define the scaling factor
scaled_threshold = 0.04 # Adjust this as needed

def is_point_within_edge(point, edge_coordinates):
    """Check if the lidar point is within the segment defined by the edge coordinates."""
    p1, p2 = np.array(edge_coordinates[0]), np.array(edge_coordinates[1])
    edge_vector = p2 - p1
    point_vector = np.array(point) - p1
    # Project point_vector onto edge_vector
    proj_length = np.dot(point_vector, edge_vector) / np.linalg.norm(edge_vector)
    # Check if the projection length is between 0 and the length of the edge_vector
    return 0 <= proj_length <= np.linalg.norm(edge_vector)

# Function to check if a point is near a boundary and also within the edge segment
def is_near_boundary_and_within_edge(point_distances):
    for entry in point_distances:
        if entry['distance'] < scaled_threshold:
            if is_point_within_edge(entry['lidar_point'], entry['edge_coordinates']):
                return True
    return False

# Applying the check to determine boundary points
boundary_points = []
for i, distances in point_line_distances.items():
    if is_near_boundary_and_within_edge(distances):
        boundary_points.append(unique_points[i])
# Visualize the results
lidar_points = np.array(unique_points)

# Dictionary to hold points by strip
strip_points = defaultdict(list)

# Adjust the loop to group points by strip
for i, distances in point_line_distances.items():
    if is_near_boundary_and_within_edge(distances):
        # Assuming each 'distances' entry has a 'camera' and 'edge_idx' to uniquely identify a strip
        for entry in distances:
            if entry['distance'] < scaled_threshold and is_point_within_edge(entry['lidar_point'], entry['edge_coordinates']):
                strip_name = f"{entry['camera']}_strip{entry['edge_idx']}"
                strip_points[strip_name].append(entry['lidar_point'])




boundary_points_np = np.array(boundary_points)

pcd_lidar = o3d.geometry.PointCloud()
pcd_boundary = o3d.geometry.PointCloud()

pcd_lidar.points = o3d.utility.Vector3dVector(lidar_points)
pcd_boundary.points = o3d.utility.Vector3dVector(boundary_points_np)

pcd_lidar.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.5, 0.5, 0.5]]), (len(lidar_points), 1)))
pcd_boundary.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1, 0, 0]]), (len(boundary_points_np), 1)))

o3d.visualization.draw_geometries([pcd_boundary, pcd_lidar], window_name="Lidar Data Visualization")

save_location = rf"C:\Users\evans\OneDrive - University of Southampton\Desktop\Year 3\Year 3 Project\Overlap_Project_Programme\pythonProject1\Images\LIDAR_OVERLAP_TESTS\Test_For_{scaled_threshold}_7"

if not os.path.exists(save_location):
    os.makedirs(save_location)

#lidar_projected_on_to_camera_dict = image_creation(seq_id, frame_id, boundary_points_np, save_location)
lidar_projected_on_to_camera_dict = image_creation(seq_id, frame_id, strip_points, save_location)

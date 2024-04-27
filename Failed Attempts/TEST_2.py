from Plot_Fustum_PointCloud import unique_points, image_creation, seq_id, frame_id, frustum_dict
from scipy.spatial.distance import euclidean
import numpy as np
import time
import open3d as o3d
from scipy.spatial import ConvexHull, Delaunay


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

def line_function(frustums_edges):
    line_functions = {}
    frustum_edges = extract_top_edges(frustums_edges)
    for cam, edges in frustum_edges.items():
        line_functions[cam] = []
        for edge in edges:
            # Convert the points from lists to numpy arrays if they aren't already
            p1 = np.array(edge[0])
            p2 = np.array(edge[1])

            # Create the line function and store it
            line_func = define_line(p1, p2)
            line_functions[cam].append(line_func)
    return line_functions


def point_to_parametric_line_dist(point, p1, p2):
    """
    Calculate the perpendicular distance from a point to a parametric line.
    The line_func is a function that returns a point on the line for a given t.
    p1 and p2 are points that define the line.
    """
    P = np.array(point)
    P1 = np.array(p1)
    P2 = np.array(p2)

    # Direction vector of the line
    line_dir = P2 - P1
    line_dir_normalized = line_dir / np.linalg.norm(line_dir)

    # Vector from P1 to the point
    P1_to_P = P - P1

    # Project vector P1_to_P onto the line direction to find the closest point on the line
    projection_length = np.dot(P1_to_P, line_dir_normalized)
    closest_point_on_line = P1 + projection_length * line_dir_normalized

    # The perpendicular distance from the point to the line is the magnitude
    # of the vector from P to the closest point on the line
    distance_vector = P - closest_point_on_line
    distance = np.linalg.norm(distance_vector)

    return distance

frustum_edges = extract_top_edges(frustum_dict)
# Initialize a structure to hold the distances of each point to each line
point_line_distances = {i: [] for i, _ in enumerate(unique_points)}

def distance_to_camera(point, camera_location):
    return np.linalg.norm(np.array(point) - np.array(camera_location))


# Define the scaling factor
scale_factor = 0.01  # Adjust this as needed


# Function to check if a point is near a boundary, now taking into account dynamic camera locations
def is_near_boundary(point_distances, frustum_edges):
    is_boundary_point = False
    for entry in point_distances:
        point = entry['coordinates']
        camera_key = entry['camera']
        edge_idx = entry['edge_idx']

        # Determine the camera location dynamically from the bottom point of the frustum edge
        camera_location = np.array(frustum_edges[camera_key][edge_idx][0])  # Use the first point as the camera location

        # Calculate the distance from the point to the camera location
        distance_from_camera = distance_to_camera(point, camera_location)
        # Scale the threshold based on the distance from the camera
        scaled_threshold = 1 + scale_factor * distance_from_camera

        if entry['distance'] < scaled_threshold:
            is_boundary_point = True
            break
    return is_boundary_point

# Iterate over each point and each line
# Initialize a structure to hold the distances and coordinates of each point to each line
point_line_distances = {i: [] for i, _ in enumerate(unique_points)}

# Iterate over each point and each line
for i, point in enumerate(unique_points):
    #print(f"{i}/{len(unique_points)}")
    for cam, edges in frustum_edges.items():
        for edge_idx, edge_points in enumerate(edges):
            p1, p2 = edge_points  # Unpack the two points defining the line

            # Calculate the distance from the current point to this line
            distance = point_to_parametric_line_dist(point, p1, p2)

            # Store the distance along with the camera, edge index, and point coordinates
            point_line_distances[i].append({
                'camera': cam,
                'edge_idx': edge_idx,
                'distance': distance,
                'coordinates': point  # Store the coordinates of the point
            })

# Applying the perspective-corrected threshold to determine boundary points
boundary_points = []
for i, distances in point_line_distances.items():
    if is_near_boundary(distances, frustum_edges):
        boundary_points.append(unique_points[i])  # Collect points that are considered boundary points

print(boundary_points)
# Assuming 'unique_points' is your entire lidar point cloud and 'boundary_points' are the points you've identified as boundaries
lidar_points = np.array(unique_points)  # Convert your lidar points list to a numpy array if it isn't already
boundary_points_np = np.array(boundary_points)  # Convert boundary points list to a numpy array

# Create Open3D point cloud objects
pcd_lidar = o3d.geometry.PointCloud()
pcd_boundary = o3d.geometry.PointCloud()

# Set points
pcd_lidar.points = o3d.utility.Vector3dVector(lidar_points)
pcd_boundary.points = o3d.utility.Vector3dVector(boundary_points_np)

# Color the entire lidar points in gray
pcd_lidar.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.5, 0.5, 0.5]]), (len(lidar_points), 1)))

# Color the boundary points in red
pcd_boundary.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1, 0, 0]]), (len(boundary_points_np), 1)))

# Combine the point clouds (optional, if you want them in the same object for some reason)
combined_pcd = pcd_lidar + pcd_boundary

for point in boundary_points:
    print(point)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd_boundary], window_name="Lidar Data Visualization", width=800, height=600, left=50, top=50)



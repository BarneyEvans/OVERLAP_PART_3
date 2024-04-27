import os

from Plot_Fustum_PointCloud import unique_points, image_creation, seq_id, frame_id, frustum_dict
from scipy.spatial.distance import euclidean
import numpy as np
import time
import open3d as o3d
from scipy.spatial import ConvexHull, Delaunay
from collections import defaultdict
from Determining_Threshold_Points import is_near_boundary_and_within_edge, distance_to_line_along_local_y, is_point_within_edge, extract_top_edges, define_line, define_local_y_axis




def distances_from_points_to_frustums(points, dict):
    frustum_edges = extract_top_edges(dict)
    point_line_distances = {i: [] for i, _ in enumerate(unique_points)}

    # Iterate over each point and each line
    for i, point in enumerate(points):
        for cam, edges in frustum_edges.items():
            for edge_idx, edge_points in enumerate(edges):
                p1, p2 = edge_points  # Unpack the two points defining the line

                # Calculate the distance from the current point to this line along the local Y-axis
                distance = distance_to_line_along_local_y(point, p1, p2)

                # Store the distance along with the camera, edge index, and local Y-axis
                point_line_distances[i].append({'camera': cam, 'edge_idx': edge_idx, 'distance': distance, 'edge_coordinates': edge_points, 'lidar_point': point})
    return point_line_distances


def create_boundary_points_list(point_distance, threshold):
    # Applying the check to determine boundary points
    boundary_points = []
    for i, distances in point_distance.items():
        if is_near_boundary_and_within_edge(distances, threshold):
            boundary_points.append(unique_points[i])
    # Visualize the results
    lidar_points = np.array(unique_points)
    return boundary_points, lidar_points


def create_bounadary_dict(point_distance, threshold):
    # Dictionary to hold points by strip
    strip_points = defaultdict(list)

    # Adjust the loop to group points by strip
    for i, distances in point_distance.items():
        if is_near_boundary_and_within_edge(distances, threshold):
            # Assuming each 'distances' entry has a 'camera' and 'edge_idx' to uniquely identify a strip
            for entry in distances:
                if entry['distance'] < threshold and is_point_within_edge(entry['lidar_point'], entry['edge_coordinates']):
                    strip_name = f"{entry['camera']}_strip{entry['edge_idx']}"
                    strip_points[strip_name].append(entry['lidar_point'])
    return strip_points


def visualise_boundary_3D(boundary_points, lidar_points):
    boundary_points_np = np.array(boundary_points)

    pcd_lidar = o3d.geometry.PointCloud()
    pcd_boundary = o3d.geometry.PointCloud()

    pcd_lidar.points = o3d.utility.Vector3dVector(lidar_points)
    pcd_boundary.points = o3d.utility.Vector3dVector(boundary_points_np)

    pcd_lidar.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.5, 0.5, 0.5]]), (len(lidar_points), 1)))
    pcd_boundary.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1, 0, 0]]), (len(boundary_points_np), 1)))

    o3d.visualization.draw_geometries([pcd_boundary, pcd_lidar], window_name="Lidar Data Visualization")




scaled_threshold = 0.04
def main():
    distances = distances_from_points_to_frustums(unique_points, frustum_dict)
    bp,lp = create_boundary_points_list(distances, scaled_threshold)
    strip_dict = create_bounadary_dict(distances, scaled_threshold)
    visualise_boundary_3D(bp,lp)
    return strip_dict


save_location = rf"C:\Users\evans\OneDrive - University of Southampton\Desktop\Year 3\Year 3 Project\Overlap_Project_Programme\pythonProject1\Images\LIDAR_OVERLAP_TESTS\Test_For_{scaled_threshold}_8"

strip_points = main()


if not os.path.exists(save_location):
    os.makedirs(save_location)

#lidar_projected_on_to_camera_dict = image_creation(seq_id, frame_id, boundary_points_np, save_location)
lidar_projected_on_to_camera_dict = image_creation(seq_id, frame_id, strip_points, save_location)

from Plot_Fustum_PointCloud import lines, unique_points, image_creation, seq_id, frame_id, frustum_dict
from scipy.spatial.distance import euclidean
import numpy as np
import time


def closest_point_on_line(point, line):
    """
    Calculate the closest point on a line to a given point in space.

    Parameters:
    point (np.array): The point in space (x, y, z).
    line (tuple): A tuple containing a point on the line (P0) and the line's direction vector (dir).

    Returns:
    np.array: The closest point on the line.
    """
    P0, dir = line
    P0 = np.array(P0)
    dir = np.array(dir)
    t = np.dot((point - P0), dir) / np.dot(dir, dir)
    closest_point = P0 + t * dir
    return closest_point


def is_point_near_line(point, line, threshold):
    """
    Determine if a point is within a threshold distance from a line.

    Parameters:
    point (np.array): The point in space (x, y, z).
    line (tuple): A tuple containing a point on the line (P0) and the line's direction vector (dir).
    threshold (float): The threshold distance.

    Returns:
    bool: True if the point is within the threshold distance from the line, False otherwise.
    """
    closest_point = closest_point_on_line(point, line)
    distance = euclidean(point, closest_point)
    return distance <= threshold


def find_lidar_points_near_lines(lidar_points, frustum_line_equations, threshold):
    lines__ = """
    Find all LiDAR points that are within a threshold distance from any of the frustum lines.

    Parameters:
    lidar_points (np.array): An array of LiDAR points (N x 3).
    frustum_line_equations (list): A list of tuples representing the line equations.
    threshold (float): The threshold distance to consider a point as being "near" a line.

    Returns:
    list: A list of LiDAR points that are near any of the frustum lines.
    """
    near_points = []
    for i, point in enumerate(lidar_points):
        print(f"{i}/{len(lidar_points)}")
        for line in frustum_line_equations:
            if is_point_near_line(point, line, threshold):
                near_points.append(point)
                break  # Only add the point once if it's near any line
    return near_points


print(lines)
time.sleep(20)

near_points = find_lidar_points_near_lines(unique_points, lines, threshold=5)
near_points = np.vstack(near_points)
path = r"/Images/Boundary_Approximation/Attempt_5"
image_creation(seq_id, frame_id, near_points, path)

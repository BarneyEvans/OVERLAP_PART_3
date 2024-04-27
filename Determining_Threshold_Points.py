import numpy as np

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

def is_point_within_edge(point, edge_coordinates):
    """Check if the LiDAR point is within the segment defined by the edge coordinates."""
    p1, p2 = np.array(edge_coordinates[0]), np.array(edge_coordinates[1])
    edge_vector = p2 - p1
    point_vector = np.array(point) - p1
    # Project point_vector onto edge_vector
    proj_length = np.dot(point_vector, edge_vector) / np.linalg.norm(edge_vector)
    # Check if the projection length is between 0 and the length of the edge_vector
    return 0 <= proj_length <= np.linalg.norm(edge_vector)

def is_near_boundary_and_within_edge(point_distances, scaled_threshold):
    """Determine if a point is near a boundary and within an edge using a scaled threshold."""
    for entry in point_distances:
        if entry['distance'] < scaled_threshold:
            if is_point_within_edge(entry['lidar_point'], entry['edge_coordinates']):
                return True
    return False

def extract_top_edges(frustums):
    """Extract the top edges of the frustums"""
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
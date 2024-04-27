import numpy as np
import matplotlib.pyplot as plt
from Overlapping_Region_Visualiser import camera_specs, cube_corners

def project_to_2d(camera_params, points):
    position, lookat, up, fov, aspect, near, far = camera_params
    # Normalize direction vectors
    forward = lookat - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)  # Correct the up vector to be orthogonal to the right and forward vectors

    # Camera matrix (not including perspective projection yet)
    camera_matrix = np.array([right, up, forward])

    # Projection transformations
    transformed_points = []
    for point in points:
        # Transform point to camera coordinates
        cam_coord = point - position
        cam_coord_in_camera_space = camera_matrix.dot(cam_coord)

        x, y, z = cam_coord_in_camera_space
        if z <= 0:  # Discard points that are behind the camera
            continue

        x = (X / Z) * near * (1 / tan(fov / 2))
        y = (Y / Z) * near * (1 / tan(fov / 2)) / aspect

        transformed_points.append((x, y))

    return np.array(transformed_points)

def plot_cube_2d(ax, corners, edges):
    if corners.size == 0:
        return
    for edge in edges:
        start, end = corners[edge[0]], corners[edge[1]]
        ax.plot([start[0], end[0]], [start[1], end[1]], c='k')

cube_edges_indices = [
    [0, 1], [1, 3], [3, 2], [2, 0],
    [4, 5], [5, 7], [7, 6], [6, 4],
    [0, 4], [1, 5], [3, 7], [2, 6]
]

for camera_name, params in camera_specs.items():
    camera_params = (params['position'], params['lookat'], params['up'], params['fov'],
                     params['aspect'], params['near'], params['far'])
    projected_corners = project_to_2d(camera_params, cube_corners)

    # Set plot limits based on the maximum extent of the points
    if projected_corners.size > 0:
        x_min, y_min = projected_corners.min(axis=0)
        x_max, y_max = projected_corners.max(axis=0)
        fig, ax = plt.subplots()
        plot_cube_2d(ax, projected_corners, cube_edges_indices)
        ax.set_aspect('equal')
        ax.set_xlim(x_min - 0.1, x_max + 0.1)
        ax.set_ylim(y_min - 0.1, y_max + 0.1)
        ax.set_title(f'2D Projection from {camera_name}')
        plt.show()

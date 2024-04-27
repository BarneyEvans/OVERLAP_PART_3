import numpy as np
import open3d as o3d


def quaternion_to_rotation_matrix(quat):
    # Convert quaternion [x, y, z, w] to a rotation matrix
    x, y, z, w = quat
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    rotation_matrix = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
    ])
    return rotation_matrix


def frustum_corners(position, rotation, h_fov, v_fov, near, far):
    rotation_matrix = quaternion_to_rotation_matrix(rotation)
    forward = rotation_matrix[:, 2]
    right = rotation_matrix[:, 0]
    up = rotation_matrix[:, 1]

    Hnear = 2 * np.tan(np.radians(v_fov) / 2) * near
    Wnear = 2 * np.tan(np.radians(h_fov) / 2) * near
    Hfar = 2 * np.tan(np.radians(v_fov) / 2) * far
    Wfar = 2 * np.tan(np.radians(h_fov) / 2) * far

    cnear = position + forward * near
    cfar = position + forward * far

    corners_near = [
        cnear + up * Hnear / 2 - right * Wnear / 2,
        cnear - up * Hnear / 2 - right * Wnear / 2,
        cnear - up * Hnear / 2 + right * Wnear / 2,
        cnear + up * Hnear / 2 + right * Wnear / 2,
    ]
    corners_far = [
        cfar + up * Hfar / 2 - right * Wfar / 2,
        cfar - up * Hfar / 2 - right * Wfar / 2,
        cfar - up * Hfar / 2 + right * Wfar / 2,
        cfar + up * Hfar / 2 + right * Wfar / 2,
    ]
    return np.array(corners_near), np.array(corners_far)


def create_frustum_lines(position, rotation, h_fov, v_fov, near, far):
    corners_near, corners_far = frustum_corners(position, rotation, h_fov, v_fov, near, far)
    points = np.vstack((corners_near, corners_far, position))
    lines = [[i, i + 4] for i in range(4)] + [[i, (i + 1) % 4] for i in range(4)] + [[i + 4, (i + 1) % 4 + 4] for i in
                                                                                     range(4)] + [[8, i + 4] for i in
                                                                                                  range(4)]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    return line_set


# Cube corners (assuming a unit cube centered at the origin)
cube_corners = np.array([
    [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
])

# Cube edges based on the corners
cube_edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7]
]

# Create a line set for the cube edges
cube_lineset = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(cube_corners),
    lines=o3d.utility.Vector2iVector(cube_edges)
)

# Paint the lineset red to represent the edges
cube_lineset.paint_uniform_color([1, 0, 0])  # Red color

# Example camera specifications with position, rotation as quaternion, h_fov, v_fov, near, far
camera_specs = {
    'Camera1': {'position': [0, 0, -2], 'rotation': [0, 0, 0, 1], 'h_fov': 60, 'v_fov': 40, 'near': 0.1, 'far': 10},
    'Camera2': {'position': [0, 0.5, -2], 'rotation': [0, 0, 0, 1], 'h_fov': 60, 'v_fov': 40, 'near': 0.1, 'far': 10},

}

# Generate frustum lines for each camera based on the specs
frustum_line_sets = [create_frustum_lines(**specs) for specs in camera_specs.values()]

# Initialize Open3D visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add the cube and frustums to the visualizer
vis.add_geometry(cube_lineset)
for frustum_lineset in frustum_line_sets:
    vis.add_geometry(frustum_lineset)

# Run the visualizer
vis.run()
vis.destroy_window()

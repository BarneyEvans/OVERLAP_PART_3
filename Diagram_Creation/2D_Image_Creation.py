import numpy as np
import matplotlib.pyplot as plt
from Overlapping_Region_Visualiser import camera_specs, cube_corners


def quaternion_to_rotation_matrix(quat):
    # Convert quaternion [x, y, z, w] to a rotation matrix
    x, y, z, w = quat
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    rotation_matrix = np.array([
        [1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy)],
        [2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx)],
        [2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy)],
    ])
    return rotation_matrix


def project_points(points, position, rotation, h_fov, v_fov, image_width, image_height):
    rotation_matrix = quaternion_to_rotation_matrix(rotation)
    translation_vector = -position
    forward_vector = rotation_matrix[:, 2]

    # Intrinsic matrix calculation
    f_x = image_width / (2 * np.tan(np.radians(h_fov) / 2))
    f_y = image_height / (2 * np.tan(np.radians(v_fov) / 2))
    c_x = image_width / 2
    c_y = image_height / 2
    intrinsic_matrix = np.array([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0, 0, 1],
    ])

    # Extrinsic matrix calculation (rotation followed by translation)
    extrinsic_matrix = rotation_matrix
    extrinsic_matrix[:, 2] *= -1  # Convert to left-handed coordinate system
    camera_space_points = np.dot(points - translation_vector, extrinsic_matrix.T)

    # Ignore points behind the camera
    camera_space_points = camera_space_points[camera_space_points[:, 2] > 0]

    # Perspective projection
    projected_points = np.dot(camera_space_points, intrinsic_matrix.T)
    projected_points /= projected_points[:, 2, np.newaxis]  # Divide by Z to get image plane coordinates

    # Normalize into pixel coordinates
    projected_points = projected_points[:, :2]
    projected_points[:, 1] = image_height - projected_points[:, 1]  # Flip Y-axis for image coordinates
    return projected_points


# Camera parameters
position = np.array([0, 0, -2])
rotation = np.array([0, 0, 0, 1])  # Identity quaternion for no rotation
h_fov = 90  # Horizontal field of view in degrees
v_fov = 90  # Vertical field of view in degrees
image_width = 800
image_height = 600

# Define cube corners (A standard unit cube centered at the origin)
cube_corners = np.array([
    [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
])

# Project the cube corners onto the 2D image plane
image_points = project_points(cube_corners, position, rotation, h_fov, v_fov, image_width, image_height)

# Plot the 2D projection
plt.figure()
plt.scatter(image_points[:, 0], image_points[:, 1], color='red')
plt.title('2D Projection of a Cube')
plt.xlabel('Pixel X')
plt.ylabel('Pixel Y')
plt.xlim(0, image_width)
plt.ylim(0, image_height)
plt.gca().invert_yaxis()  # Invert the Y-axis to match image coordinate system
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

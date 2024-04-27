from once import ONCE
import numpy as np
import open3d as o3d
from Frustum import calculate_frustum_corners
import cv2
import os
import matplotlib.pyplot as plt


dataset = ONCE(r'C:\Users\evans\OneDrive - University of Southampton\Desktop\Year 3\Year 3 Project\Full_DataSet')
seq_id = "000076"
frame_id = "1616343528200"
cam_names = ["cam01", "cam03", "cam05", "cam06", "cam07", "cam08", "cam09"]
seq_id = "000275"
frame_id = "1619040581398"
#cam_names = ["cam07", "cam08"]
new_cam_intrinsics_dict, old_intrinsic_dict, extrinsic_dict = dataset.get_vital_info(seq_id, frame_id)
img_buf_dict, unique_points, colours, image_points = dataset.project_lidar_to_image_with_colour(seq_id, frame_id)

camera_colors = {
    'cam01': [255, 0, 0],  # Red
    'cam03': [0, 255, 0],  # Green
    'cam05': [0, 0, 255],  # Blue
    'cam06': [255, 255, 0],  # Yellow
    'cam07': [255, 0, 255],  # Magenta
    'cam08': [0, 255, 255],  # Cyan
    'cam09': [255, 165, 0]  # Orange
}


def visualize_coloured_frustums_with_point_cloud(lidar_points, point_colours, frustums, output_bool):
    if len(lidar_points) != len(point_colours):
        raise ValueError("The number of LiDAR points must match the number of colour entries.")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_points)
    for i, colour in enumerate(point_colours):
        if np.array_equal(colour, [120, 120, 120]):
            point_colours[i] = [0, 0, 0]
    pcd.colors = o3d.utility.Vector3dVector(point_colours)
    geometries = [pcd]
    frustum_line_equations = []

    # Add each frustum to the visualization and collect line equations
    for points_lidar, colour in frustums:
        # Define the edge pairs for the full frustum visualization
        full_edge_pairs = [[i, i + 4] for i in range(4)] + \
                          [[i, (i + 1) % 4] for i in range(4)] + \
                          [[i + 4, (i + 1) % 4 + 4] for i in range(4)]

        # Define the line set for full frustum visualization
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points_lidar),
            lines=o3d.utility.Vector2iVector(full_edge_pairs),
        )
        line_set.colors = o3d.utility.Vector3dVector([colour for _ in full_edge_pairs])
        geometries.append(line_set)

        # Focus only on the top vertical edge pairs for line equation extraction
        top_edge_pairs = [[2, 6], [3, 7]]
        for start_idx, end_idx in top_edge_pairs:
            P0 = points_lidar[start_idx]
            P1 = points_lidar[end_idx]
            direction = np.array(P1) - np.array(P0)
            frustum_line_equations.append((P0, direction))

    if output_bool:
        # Draw all geometries together
        o3d.visualization.draw_geometries(geometries)

    # Return the line equations for the top vertical edges
    return frustum_line_equations






def image_creation(seq, frame, frust_ums, save_location):
    if isinstance(frust_ums, dict):
        img_buf_dict = dataset.project_own_lidar_to_image_remove_noise(seq, frame, frust_ums)
    else:
        img_buf_dict, lidar_projected_on_to_camera_dict = dataset.project_own_lidar_to_image(seq, frame, frust_ums)
    for cam_name, img_buf in img_buf_dict.items():
        cv2.imwrite(os.path.join(save_location, f"{cam_name}_{seq}_{frame}.jpg"),
                    cv2.cvtColor(img_buf, cv2.COLOR_BGR2RGB))
    return img_buf_dict



frustums_and_colours = []
frustum_dict = {}
#frustums = np.array([]).reshape(0, 3)  # Initialize an empty array with the right shape
for cam_name in cam_names:
    points = calculate_frustum_corners(new_cam_intrinsics_dict[cam_name], extrinsic_dict[cam_name], 0.01, 150)
    colour = camera_colors[cam_name]
    frustum_dict[cam_name] = points
    #frustums = np.vstack((frustums, points)) if frustums.size else points  # Stack arrays vertically
    frustums_and_colours.append((points, colour))


lines = visualize_coloured_frustums_with_point_cloud(unique_points, colours, frustums_and_colours, False)
print(unique_points)

#image_creation(seq_id, frame_id, frustums, 'images/Frustum_On_Image/lidar_project_{}_{}.jpg')



from once import ONCE
import open3d as o3d
import numpy as np


dataset = ONCE(r'C:\Users\evans\OneDrive - University of Southampton\Desktop\Year 3\Year 3 Project\Full_DataSet')
seq_id = "000076"
frame_id = "1616343528200"
img_buf_dict, dictionary, unique_points, colors = dataset.project_lidar_to_image_with_colour(seq_id, frame_id)
#img_buf_dict, unique_points, colors = dataset.project_lidar_to_image_with_color_v2(seq_id, frame_id)

# Assuming 'used_points' is a list of NumPy arrays, first convert it to a single NumPy array
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(unique_points)
pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize colors for visualization
o3d.visualization.draw_geometries([pcd], window_name="LiDAR Points with Camera-based Colors")
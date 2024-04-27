from once import ONCE
import numpy as np
import open3d as o3d
from Frustum import calculate_frustum_corners
import cv2
import time

dataset = ONCE(r'C:\Users\evans\OneDrive - University of Southampton\Desktop\Year 3\Year 3 Project\Full_DataSet')
seq_id = "000076"
frame_id = "1616343528200"
cam_names = ["cam01", "cam03", "cam05", "cam06", "cam07", "cam08", "cam09"]

new_cam_intrinsics_dict, old_intrinsic_dict, extrinsic_dict = dataset.get_vital_info(seq_id, frame_id)
img_buf_dict, unique_points, colours, image_points_colour = dataset.project_lidar_to_image_with_colour(seq_id, frame_id)

def get_boundary_points(imc):
    cam_overlap_points = {
        'cam01': [],
        'cam03': [],
        'cam05': [],
        'cam06': [],
        'cam07': [],
        'cam08': [],
        'cam09': []
    }

    # Iterate over each camera and its points
    for cam, points in imc.items():
        # Add points where color is [120, 120, 120] to the cam_overlap_points under the respective camera
        for point, color in points.items():
            if color == [120, 120, 120]:
                cam_overlap_points[cam].append(point)

    # Dictionary to hold boundary points for each camera
    boundary_points = {}

    # Calculate boundary points from the overlapping points
    for cam, points in cam_overlap_points.items():
        if points:
            # Extract x and y coordinates
            X_values = [p[0] for p in points]
            Y_values = [p[1] for p in points]

            # Calculate minimum and maximum x for each unique y coordinate
            min_tuples = {y: min(x for x, y1 in points if y1 == y) for y in set(Y_values)}
            max_tuples = {y: max(x for x, y1 in points if y1 == y) for y in set(Y_values)}

            # Combine min and max x values for each y into a tuple and store it in the boundary_points dictionary
            boundary_points[cam] = [(min_x, y, max_x) for y, min_x in min_tuples.items() if y in max_tuples for max_x in
                                    [max_tuples[y]]]

    return boundary_points





points = get_boundary_points(image_points_colour)
print(points.keys())
images = dataset.plot_image_points(seq_id, frame_id, points)
for cam_name, img_buf in images.items():
    cv2.imwrite('images/Boundary_Points/Boundary_Points_{}_{}.jpg'.format(cam_name, frame_id),
                cv2.cvtColor(img_buf, cv2.COLOR_BGR2RGB))



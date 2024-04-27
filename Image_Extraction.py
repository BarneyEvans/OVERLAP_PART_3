from once import ONCE
import numpy as np
import open3d as o3d
from Frustum import calculate_frustum_corners
import cv2
import os

dataset = ONCE(r'C:\Users\evans\OneDrive - University of Southampton\Desktop\Year 3\Year 3 Project\Full_DataSet')
seq_id = "000076"
frame_id = "1616343528200"
output_directory = r"C:\Users\evans\OneDrive - University of Southampton\Desktop\Year 3\Year 3 Project\Overlap_Project_Programme\pythonProject1\Images\Normal_Images"


images, _ = dataset.undistort_image_v2(seq_id, frame_id)
points_img_dict = {}
for cam_no, cam_name in enumerate(dataset.__class__.camera_names):
    img_buf = images[cam_no]
    points_img_dict[cam_name] = img_buf


for cam_name, img_buf in points_img_dict.items():
    image_path = os.path.join(output_directory, "{}_{}.jpg".format(cam_name, frame_id))
    cv2.imwrite(image_path, cv2.cvtColor(img_buf, cv2.COLOR_BGR2RGB))
    print(f"Saved: {image_path}")
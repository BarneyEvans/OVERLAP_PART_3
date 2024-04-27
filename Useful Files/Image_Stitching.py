
import numpy as np
import os
import cv2 as cv


img1 = cv.imread(r"C:\Users\evans\OneDrive - University of Southampton\Desktop\Year 3\Year 3 Project\Overlap_Project_Programme\pythonProject1\Images\LIDAR_OVERLAP_TESTS\Test_8\cam07_000076_1616343528200.jpg")
img2 = cv.imread(r"C:\Users\evans\OneDrive - University of Southampton\Desktop\Year 3\Year 3 Project\Overlap_Project_Programme\pythonProject1\Images\LIDAR_OVERLAP_TESTS\Test_8\cam08_000076_1616343528200.jpg")

side_by_side1 = np.concatenate((img1, img2), axis=1)
output_dir = r'C:\Users\evans\OneDrive - University of Southampton\Desktop\Year 3\Year 3 Project\Overlap_Project_Programme\pythonProject1\Images\Stitched_Images'
side_by_side1_path = os.path.join(output_dir, 'side_by_side1.jpg')
cv.imwrite(side_by_side1_path, side_by_side1)



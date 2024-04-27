from Plot_Fustum_PointCloud import unique_points, image_creation, seq_id, frame_id, frustum_dict
from scipy.spatial.distance import euclidean
import numpy as np
import time
import open3d as o3d
from scipy.spatial import ConvexHull, Delaunay


def extract_top_edges(frustums):
    top_edges = {}
    for camera, points in frustums.items():
        top_edges[camera] = [
            (points[0], points[4]),
            (points[1], points[5])
        ]
    return top_edges


def y_threshold_scaling(distance_from_camera, base_threshold, scaling_factor):
    return base_threshold + (distance_from_camera * scaling_factor)


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def box_creation(top_edges, z_height, z_base, base_y_threshold, scaling_factor):
    boxes = {}
    for camera, edges in top_edges.items():
        camera_boxes = []
        for edge in edges:
            p1, p2 = edge
            local_x_axis = normalize(np.array(p2) - np.array(p1))
            global_z_axis = np.array([0, 0, 1])  # Assuming Z-axis up globally
            local_y_axis = normalize(np.cross(global_z_axis, local_x_axis))

            # Calculate distances from camera to start and end points of the edge
            distance_start = np.linalg.norm(np.array(p1))
            distance_end = np.linalg.norm(np.array(p2))

            # Calculate dynamic y-threshold at the start and end of the edge
            y_threshold_start = y_threshold_scaling(distance_start, base_y_threshold, scaling_factor)
            y_threshold_end = y_threshold_scaling(distance_end, base_y_threshold, scaling_factor)

            # Generate the vertices of the box dynamically based on the scaled Y-threshold
            vertices = []
            for dx in [0, 1]:  # From base (0) to top (1) of edge
                # Interpolate the dynamic y-threshold for the current vertex
                dy_threshold = np.interp(dx, [0, 1], [y_threshold_start, y_threshold_end])
                for dy in [-dy_threshold, dy_threshold]:  # Extend in local Y direction
                    for dz in [z_base, z_height]:  # Extend in global Z direction
                        vertex = (np.array(p1) + dx * (np.array(p2) - np.array(p1)) +
                                  dy * local_y_axis + dz * global_z_axis)
                        vertices.append(vertex)
            camera_boxes.append(vertices)
        boxes[camera] = camera_boxes
    return boxes


top_edges = extract_top_edges(frustum_dict)
z_height = 0
z_base = -50
y_left = 0.2
y_right = 0.2
boxes = box_creation(top_edges, z_height, z_base, 0.05, 0.01)

print(boxes)


def filter_points_by_aabb(original_point_cloud, boxes):
    """
    Filters points to include only those within the specified AABBs.
    """
    filtered_points = []
    for i, point in enumerate(original_point_cloud):
        point = np.array(point)
        in_any_box = False
        print(f"{i}/{len(original_point_cloud)}")

        for camera, camera_boxes in boxes.items():
            for box in camera_boxes:
                # Calculate the min and max for each dimension
                min_x, min_y, min_z = np.min(box, axis=0)
                max_x, max_y, max_z = np.max(box, axis=0)

                # Check if the point lies within the min and max bounds
                if ((min_x <= point[0] <= max_x) and
                        (min_y <= point[1] <= max_y) and
                        (min_z <= point[2] <= max_z)):
                    in_any_box = True
                    break
            if in_any_box:
                break

        if in_any_box:
            filtered_points.append(point)

    return np.array(filtered_points)


def plot_point_cloud_with_boxes(original_point_cloud, filtered_point_cloud, boxes):
    """
    Plots the original and filtered point clouds with bounding boxes using open3d.
    """
    # Convert numpy arrays to Open3D point cloud format
    pcd_original = o3d.geometry.PointCloud()
    pcd_filtered = o3d.geometry.PointCloud()

    pcd_original.points = o3d.utility.Vector3dVector(original_point_cloud)
    pcd_filtered.points = o3d.utility.Vector3dVector(filtered_point_cloud)

    # Set colors for the point clouds: red for original, green for filtered
    pcd_original.paint_uniform_color([1, 0, 0])  # Red
    pcd_filtered.paint_uniform_color([0, 1, 0])  # Green

    # Prepare the visualizer
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(pcd_original)
    visualizer.add_geometry(pcd_filtered)

    # Function to draw bounding boxes
    def draw_bounding_box(box):
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]  # Indices of line ends for a cuboid

        colors = [[1, 0, 0] for i in range(len(lines))]  # Red lines
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.array(box)),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    # Add bounding boxes to the visualizer
    for camera, camera_boxes in boxes.items():
        for box in camera_boxes:
            box_visual = draw_bounding_box(box)
            visualizer.add_geometry(box_visual)

    # Update the view and render until the window is closed
    visualizer.run()
    visualizer.destroy_window()

#filtered_point_cloud = filter_points_by_aabb(unique_points, boxes)
#plot_point_cloud_with_boxes(unique_points, filtered_point_cloud, boxes)


import time

#LIBRARY IMPORTS
import numpy as np
from scipy.spatial.transform import Rotation as R

#ROS IMPORTS
import rospy
import std_msgs.msg
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# CUSTOM IMPORTS
from pcdet.models import load_data_to_gpu

def clear_marker_array(publisher):
    """
    Clear previous bbox markers
    """
    bbox_markers = MarkerArray()
    bbox_marker = Marker()
    bbox_marker.id = 0
    bbox_marker.ns = "delete_markerarray"
    bbox_marker.action = Marker.DELETEALL

    bbox_markers.markers.append(bbox_marker)
    publisher.publish(bbox_markers)

def create_3d_bbox_marker(cx, cy, cz, l, w, h, roll, pitch, yaw,
                          frame_id, time_stamp, namespace='', marker_id=0,
                          r=1.0, g=0.0, b=0.0, a=1.0, scale=0.1):
    """
    Create a 3D bounding box marker

    Args:
        cx, cy, cz: center of the bounding box
        l, w, h: length, width, and height of the bounding box
        roll, pitch, yaw: orientation of the bounding box
        frame_id: frame id of header msgs
        time_stamp: time stamp of header msgs
        namespace: namespace of the bounding box
        marker_id: id of the bounding box
        scale: scale of line width
        r, g, b, a: color and transparency of the bounding box
    
    Returns:
        marker: a 3D bounding box marker
    """

    # Create transformation matrix from the given roll, pitch, yaw
    rot_mat = R.from_euler("xyz", [roll, pitch, yaw], degrees=False).as_matrix()

    # Half-length, half-width, and half-height
    hl, hw, hh = l / 2.0, w / 2.0, h / 2.0

    # Define 8 corners of the bounding box in the local frame
    local_corners = np.array([
        [hl, hw, hh],  [hl, hw, -hh],  [hl, -hw, hh],  [hl, -hw, -hh],
        [-hl, hw, hh], [-hl, hw, -hh], [-hl, -hw, hh], [-hl, -hw, -hh]
    ]).T

    # Transform corners to the frame
    frame_corners = rot_mat.dot(local_corners)
    frame_corners += np.array([[cx], [cy], [cz]])
    
    # Define lines for the bounding box (each line by two corner indices)
    lines = [
        [0, 1], [0, 2], [1, 3], [2, 3],
        [4, 5], [4, 6], [5, 7], [6, 7],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]

    # Create the LineList marker
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = time_stamp
    marker.ns = namespace
    marker.id = marker_id
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.scale.x = scale
    marker.color.a = a
    marker.color.r = r 
    marker.color.g = g
    marker.color.b = b 
    marker.pose.orientation.w = 1.0
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.lifetime = rospy.Duration(0)

    for start, end in lines:
        start_pt = Point(*frame_corners[:, start])
        end_pt = Point(*frame_corners[:, end])
        marker.points.extend([start_pt, end_pt])

    return marker

def pcnp_to_datadict(pc_np, dataloader, frame_id=0):
    #2 Predict boxes
    input_dict = {
        'points': pc_np,
        'frame_id': frame_id,    
    }

    data_dict = dataloader.prepare_data(input_dict)
    data_dict = dataloader.collate_batch([data_dict])
    load_data_to_gpu(data_dict)

    return data_dict

def visualize_3d(model, dataloader, pc_msg, bbox_3d_pub, color_map, logger=None):

    #1 Prepare pc for model (Optional coordinate conversion if necessary)
    lidar_frame = pc_msg.header.frame_id
    lidar_ts    = pc_msg.header.stamp # ROS timestamp

    pc_data = pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True)
    pc_list = list(pc_data)
    pc_np = np.array(pc_list, dtype=np.float32)
    data_dict = pcnp_to_datadict(pc_np, dataloader, frame_id=pc_msg.header.seq)
    
    #2 Perform model inference
    inference_start_time = time.time()
    pred_dicts, _ = model.forward(data_dict)
    if logger:
        inference_time = time.time() - inference_start_time
        logger.info(f'VISUALIZE_3D: Inference time {inference_time}')
    
    #3 Publish detections
    clear_marker_array(bbox_3d_pub)
    bbox_3d_markers = MarkerArray()

    pred_boxes      = pred_dicts[0]['pred_boxes'].detach().cpu().numpy()
    pred_labels     = pred_dicts[0]['pred_labels'].detach().cpu().numpy()
    pred_cls_scores = pred_dicts[0]['pred_cls_scores'].detach().cpu().numpy()

    for bbox_idx in range(len(pred_boxes)):
        pred_label  = pred_labels[bbox_idx]
        pred_score  = pred_cls_scores[bbox_idx]
        bbox_3d     = pred_boxes[bbox_idx]

        if pred_score < 0.4:
            continue
        bbox3d_xyz      = bbox_3d[0:3]
        bbox3d_lwh      = bbox_3d[3:6]
        axis_angles     = np.array([0, 0, bbox_3d[6] + 1e-10])
        bbox3d_angle    = R.from_rotvec(axis_angles, degrees=False).as_euler('xyz', degrees=False)
        instance_id     = bbox_idx # Not enabled across frames for obj det task

        bbox_3d_color_rgb = color_map[pred_label-1]
        bbox_3d_color = (
            bbox_3d_color_rgb[0], bbox_3d_color_rgb[1], bbox_3d_color_rgb[2], 1
        )
        
        bbox_marker = create_3d_bbox_marker(
            bbox3d_xyz[0], bbox3d_xyz[1], bbox3d_xyz[2],
            bbox3d_lwh[0], bbox3d_lwh[1], bbox3d_lwh[2],
            bbox3d_angle[0], bbox3d_angle[1], bbox3d_angle[2],
            lidar_frame, lidar_ts, str(instance_id),
            instance_id,
            *bbox_3d_color,
        )
        bbox_3d_markers.markers.append(bbox_marker)
    
    bbox_3d_pub.publish(bbox_3d_markers)


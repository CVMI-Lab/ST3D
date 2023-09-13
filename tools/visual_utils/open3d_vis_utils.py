"""
open3d visualization tool box
Written by Arthur King Zhang
All rights preserved from 2023 - present.
"""
import open3d as o3d
import torch

import time
import copy
import numpy as np
import cv2
import os
from os.path import join

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]

MAX_LINE_SEGMENTS = 2500
BACKGROUND_COLOR    = np.ones(3)
POINT_COLOR         = np.zeros((1,3))+0.4

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2

"""BEGIN LINE MESH OBJECT"""
class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.25, cylinder_segments=[], update_idx=0):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius

        # Create dummy values for o3d visualizer
        self.update_cylinder_segments = len(cylinder_segments)>0
        self.cylinder_segments = cylinder_segments
        self.new_line_segs = 0
        self.update_idx = update_idx

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), 
                    center=cylinder_segment.get_center())

            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            if self.update_cylinder_segments:
                self.cylinder_segments[self.update_idx+i].vertices  = cylinder_segment.vertices
                self.cylinder_segments[self.update_idx+i].triangles = cylinder_segment.triangles
                self.cylinder_segments[self.update_idx+i].paint_uniform_color(color)
            else:
                self.cylinder_segments.append(cylinder_segment)

        self.new_line_segs += line_segments_unit.shape[0]

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)
"""END LINE MESH OBJECT"""

def build_pcd(points, o3d_pts):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    points = points[:, :3]

    # pts = o3d.geometry.PointCloud()
    o3d_pts.points = o3d.utility.Vector3dVector(points)
    point_colors = np.repeat(POINT_COLOR, points.shape[0], axis=0)
    o3d_pts.colors = o3d.utility.Vector3dVector(point_colors)

    return o3d_pts

def build_bbox(bboxes, o3d_geos, color_map, bbox_radius=0.05):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
    if bboxes.ndim == 3:
        bboxes = bboxes.squeeze(0)

    num_bbox_line_meshes_used = 0 
    bbox_labels     = bboxes[:, -1].astype(int)
    for i in range(bboxes.shape[0]):
        line_set, box3d = translate_boxes_to_o3d_instance(bboxes[i])
        bbox_corners_np = np.asarray(box3d.get_box_points())
        
        bbox_label_idx = bbox_labels[i]-1
        assert bbox_label_idx>=0 and bbox_label_idx<len(color_map), \
            f'Bad color map idx available {len(color_map)}, indexed {bbox_label_idx}'
        bbox_color = color_map[bbox_label_idx] # offset one idx

        lines = [   [0, 1], [0, 2], [2, 7], [1, 7], 
                    [3, 5], [3, 6], [5, 4], [6, 4],
                    [0, 3], [1, 6], [2, 5], [7, 4]]
        
        # Workaround for open3d line width setting
        line_mesh1 = LineMesh(bbox_corners_np, lines, bbox_color, radius=bbox_radius, cylinder_segments=o3d_geos, update_idx=num_bbox_line_meshes_used)
        o3d_geos = line_mesh1.cylinder_segments
        new_bbox_line_segs = line_mesh1.new_line_segs
        total_bbox_line_meshes = num_bbox_line_meshes_used + new_bbox_line_segs
        # bbox_line_meshes[num_bbox_line_meshes_used:total_bbox_line_meshes] = line_mesh1.cylinder_segments[]

        num_bbox_line_meshes_used = total_bbox_line_meshes
        
        # bbox_line_meshes.append(line_mesh1.geoms)
    
    return o3d_geos

def gen_bbox_colormap_from_stats(stat_path, frame, data_dict):
    """
    Colors true positive boxes blue, false negatives red

    Match data_dict gt_boxes with boxes in stat_path for get correct idx to modify
    """
    from pcdet.utils import box_utils, calibration_kitti
    np.set_printoptions(precision=3, suppress=True)
    sample_id = data_dict['frame_id'][0]

    # print("num bboxes ", data_dict['gt_boxes'].shape)

    tp_path = join(stat_path, str(sample_id), "tp.txt")
    fn_path = join(stat_path, str(sample_id), "fn.txt")

    # In format: x y z l h w r
    tp_bbox_dims = np.loadtxt(tp_path, dtype=float).reshape(-1, 7)
    fn_bbox_dims = np.loadtxt(fn_path, dtype=float).reshape(-1, 7)

    # print("tp_indices ", tp_indices.shape)
    # print("fn_indices ", fn_indices.shape)

    gt_boxes = data_dict['gt_boxes'].cpu().numpy()
    num_gt_boxes = data_dict['gt_boxes'].shape[1]
    bbox_color_map = [(0,0,0)]*num_gt_boxes # black is default bbox color

    calib = data_dict['calib'][0]
    gt_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(gt_boxes[0, :, :-1], calib)
    gt_boxes_camera[:, 1] += 0.8 # Adjust based on shift coor amount

    for bbox_idx in range(len(bbox_color_map)):
        bbox_dim = gt_boxes_camera[bbox_idx]

        bbox_tp_errors, closest_tp_error_idx = None, None
        if tp_bbox_dims.shape[0] > 0:
            bbox_tp_errors = np.linalg.norm(tp_bbox_dims - bbox_dim, axis=-1)
            closest_tp_error_idx = np.argmin(bbox_tp_errors, axis=-1)

        bbox_fn_errors, closest_fn_error_idx = None, None
        if fn_bbox_dims.shape[0] > 0:
            bbox_fn_errors = np.linalg.norm(fn_bbox_dims - bbox_dim, axis=-1)
            closest_fn_error_idx = np.argmin(bbox_fn_errors, axis=-1)

        if closest_tp_error_idx is None and closest_fn_error_idx is None:
            continue
    
        if closest_tp_error_idx is None:
            is_box_tp = False
        elif closest_fn_error_idx is None:
            is_box_tp = True
        else:   
            is_box_tp = bbox_tp_errors[closest_tp_error_idx] < bbox_fn_errors[closest_fn_error_idx]

        if is_box_tp:
            bbox_color_map[bbox_idx] = (0, 0, 1)
        else:
            bbox_color_map[bbox_idx] = (1, 0, 0)
    return bbox_color_map

# multi frame drawing
def visualize_3d(
                dataloader, model, logger, color_map,
                # front=[0.01988731276770269, 0.9464119395195808, -0.32234908953751495],
                # lookat=[0.94584656,  3.44503021, 44.89602829],
                # up=[-0.015490803158388902, 0.32266582920794568, 0.94638617787827872],
                # zoom=0.029999999999999988,
                front=[-0.02107764, -0.94638618,  0.32234909],
                lookat=[1.84421158,  43.58592222, -11.19296577],
                up=[0.01382763, -0.32266583, -0.94641194],
                zoom=935.3074360871938,
                window_width=1920, window_height=1080,
                draw_origin=True,
                save_vid_filename="",
                show_gt=True,
                view_frame=0,
                stat_path=None
                ):
    from pcdet.models import build_network, load_data_to_gpu
    def reset_line_mesh_list(line_mesh_list):
        template_mesh_object = o3d.geometry.TriangleMesh.create_cylinder(0.01, 0.01) # bbox_radius, line length
        for line_mesh_idx in range(len(line_mesh_list)):
            line_mesh_list[line_mesh_idx].vertices = template_mesh_object.vertices
            line_mesh_list[line_mesh_idx].triangles = template_mesh_object.triangles
        return line_mesh_list

    def print_cam_params(view_control):
        # Get the camera parameters
        camera_params = view_control.convert_to_pinhole_camera_parameters()

        # Extract camera information
        zoom_level = camera_params.intrinsic.get_focal_length()[0]  # Focal length in pixels
        front = camera_params.extrinsic[:3, 2]  # "front" vector
        lookat = camera_params.extrinsic[:3, 3]  # "lookat" vector
        up = camera_params.extrinsic[:3, 1]  # "up" vector

        # Print extracted information
        print("Zoom Level:", zoom_level)
        print("Front Vector:", front)
        print("Lookat Vector:", lookat)
        print("Up Vector:", up)


    vis = o3d.visualization.Visualizer()
    vis.create_window(width=window_width, height=window_height)
    vis.get_render_option().point_size = 3.0

    ctr = vis.get_view_control()
    opt = vis.get_render_option()
    opt.background_color = np.ones(3)

    # draw origin
    if draw_origin:
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    save_as_vid = len(save_vid_filename)>0
    output_video_dir = None
    output_video_path = None
    if save_as_vid:
        output_video_dir = f'{os.getcwd()}/output_videos_3D'
        os.makedirs(output_video_dir, exist_ok=True)
        output_video_path = f'{output_video_dir}/{save_vid_filename}'
        out = cv2.VideoWriter(output_video_path,
                                cv2.VideoWriter_fourcc(*'DIVX'),
                                20, (window_width, window_height))

    pcd_geo = o3d.geometry.PointCloud()
    bbox_mesh_object = o3d.geometry.TriangleMesh.create_cylinder(0.01, 0.01) # bbox_radius, line length
    gt_bbox_geos = [copy.deepcopy(bbox_mesh_object) for _ in range(MAX_LINE_SEGMENTS)]
    dt_bbox_geos = [copy.deepcopy(bbox_mesh_object) for _ in range(MAX_LINE_SEGMENTS)]

    #Set ground truth bounding box color maps
    if model is not None:
        gt_bbox_color_map = [[1, 0, 0]] * len(color_map) # Red for all classes
    else:
        gt_bbox_color_map = color_map

    with torch.no_grad():
        for idx, data_dict in enumerate(dataloader):
            if not save_as_vid:
                if idx<view_frame:
                    continue
                elif idx > view_frame:
                    break
            logger.info(f'Visualizing sample index: \t{idx + 1}')
            data_dict = dataloader.collate_batch([data_dict])
            load_data_to_gpu(data_dict)

            # Reset all line mesh settings from previous frame
            gt_bbox_geos = reset_line_mesh_list(gt_bbox_geos)
            dt_bbox_geos = reset_line_mesh_list(dt_bbox_geos)

            if model is not None:
                pred_dicts, _ = model.forward(data_dict)
                pred_boxes = torch.hstack((pred_dicts[0]['pred_boxes'], pred_dicts[0]['pred_labels'].reshape(-1, 1)))
                dt_bbox_color_map = color_map
                dt_bbox_geos = build_bbox(pred_boxes, dt_bbox_geos, dt_bbox_color_map)

            if show_gt and 'gt_boxes' in data_dict.keys():
                if stat_path is not None: # generate unique color map for dt boxes 
                    gt_bbox_color_map = gen_bbox_colormap_from_stats(stat_path, idx, data_dict)
                    gt_bbox_geos = build_bbox(data_dict['gt_boxes'], gt_bbox_geos, gt_bbox_color_map, use_stat_color_map=True)
                else:
                    gt_bbox_geos = build_bbox(data_dict['gt_boxes'], gt_bbox_geos, gt_bbox_color_map)

            pcd_geo = build_pcd(data_dict['points'][:, 1:], pcd_geo)

            if idx==0 or not save_as_vid:                
                vis.add_geometry(pcd_geo)

                for g in gt_bbox_geos:
                    vis.add_geometry(g)

                for g in dt_bbox_geos:
                    vis.add_geometry(g)

                ctr.set_zoom(zoom)
                ctr.set_up(up)
                ctr.set_lookat(lookat)
                ctr.set_front(front)
            else:
                vis.update_geometry(pcd_geo)

                for g in gt_bbox_geos:
                    vis.update_geometry(g)

                for g in dt_bbox_geos:
                    vis.update_geometry(g)

                vis.poll_events()
                vis.update_renderer()
            print("Running loop")
            if not save_as_vid:
                print("Not saving as vid")
                vis.run()
                vis.destroy_window()

            time.sleep(0.1) # 2 hz

            if save_as_vid:
                vis.capture_screen_image(f'{output_video_dir}/temp.jpg', do_render=True)
                temp_img_np = cv2.resize(cv2.imread(f'{output_video_dir}/temp.jpg'), (window_width, window_height))

                text = f'Frame {idx}'
                org = (10, 60)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 2
                color = (255, 0, 0)
                thickness = 2
                out.write(cv2.putText(temp_img_np, text, org, font, fontScale, 
                    color, thickness, cv2.LINE_AA, False))

            # Uncomment to compute viewpoint parameters
            # print_cam_params(ctr)

    if save_as_vid:
        out.release()
        os.remove(f"{output_video_dir}/temp.jpg")
        logger.info(f'3D Visualization video saved: {output_video_path}')
    logger.info(f'Completed with {idx} frames visualized.')
    vis.destroy_window()

# single frame drawing
def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True, idx=0):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = o3d.geometry.PointCloud()
    pts.points = o3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = o3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = o3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        if gt_boxes.ndim == 3:
            gt_boxes = gt_boxes.squeeze(0)
        vis = draw_box(vis, gt_boxes, (1, 0, 0))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()


def translate_boxes_to_o3d_instance(gt_boxes, line_set=None):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

    if line_set is None:
        line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None, isfirstrun=True):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_o3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        if isfirstrun:
            vis.add_geometry(line_set)
        else: 
            vis.update_geometry(line_set)
        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis
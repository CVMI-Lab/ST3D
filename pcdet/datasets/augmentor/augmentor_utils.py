import torch
import numpy as np
import numba
import copy
from ...utils import common_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...ops.iou3d_nms import iou3d_nms_utils

import warnings
try:
    from numba.errors import NumbaPerformanceWarning
    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
except:
    pass


def random_flip_along_x(gt_boxes, points):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]
    return gt_boxes, points


def random_flip_along_y(gt_boxes, points):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]
    return gt_boxes, points


def global_rotation(gt_boxes, points, rot_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
    gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation
    return gt_boxes, points


def global_scaling(gt_boxes, points, scale_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    return gt_boxes, points


def global_sampling(gt_boxes, points, gt_boxes_mask, sample_ratio_range, prob):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C)
        gt_boxes_mask: (N), boolen mask for gt_boxes
        sample_ratio_range: [min, max]. ratio to keep points remain.
        prob: prob to dentermine whether sampling this frame

    Returns:

    """
    if np.random.uniform(0, 1) > prob:
        return gt_boxes, points, gt_boxes_mask

    num_points = points.shape[0]
    sample_ratio = np.random.uniform(sample_ratio_range[0], sample_ratio_range[1])
    remain_points_num = int(num_points * sample_ratio)

    # shuffle points
    shuffle_idx = np.random.permutation(points.shape[0])
    points = points[shuffle_idx]

    # sample points
    points = points[:remain_points_num]

    # mask empty gt_boxes
    num_points_in_gt = roiaware_pool3d_utils.points_in_boxes_cpu(
        torch.from_numpy(points[:, :3]),
        torch.from_numpy(gt_boxes[:, :7])
    ).numpy().sum(axis=1)

    mask = (num_points_in_gt >= 1)
    gt_boxes_mask = gt_boxes_mask & mask
    return gt_boxes, points, gt_boxes_mask


def scale_pre_object(gt_boxes, points, gt_boxes_mask, scale_perturb, num_try=50):
    """
    uniform sacle object with given range
    Args:
        gt_boxes: (N, 7) under unified coordinates
        points: (M, 3 + C) points in lidar
        gt_boxes_mask: (N), boolen mask for
        scale_perturb:
        num_try:
    Returns:
    """
    num_boxes = gt_boxes.shape[0]
    if not isinstance(scale_perturb, (list, tuple, np.ndarray)):
        scale_perturb = [-scale_perturb, scale_perturb]

    # boxes wise scale ratio
    scale_noises = np.random.uniform(scale_perturb[0], scale_perturb[1], size=[num_boxes, num_try])
    for k in range(num_boxes):
        if gt_boxes_mask[k] == 0:
            continue

        scl_box = copy.deepcopy(gt_boxes[k])
        scl_box = scl_box.reshape(1, -1).repeat([num_try], axis=0)
        scl_box[:, 3:6] = scl_box[:, 3:6] * scale_noises[k].reshape(-1, 1).repeat([3], axis=1)

        # detect conflict
        # [num_try, N-1]
        if num_boxes > 1:
            self_mask = np.ones(num_boxes, dtype=np.bool_)
            self_mask[k] = False
            iou_matrix = iou3d_nms_utils.boxes_bev_iou_cpu(scl_box, gt_boxes[self_mask])
            ious = np.max(iou_matrix, axis=1)
            no_conflict_mask = (ious == 0)
            # all trys have conflict with other gts
            if no_conflict_mask.sum() == 0:
                continue

            # scale points and assign new box
            try_idx = no_conflict_mask.nonzero()[0][0]
        else:
            try_idx = 0

        point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(
            points[:, 0:3],np.expand_dims(gt_boxes[k], axis=0)).squeeze(0)

        obj_points = points[point_masks > 0]
        obj_center, lwh, ry = gt_boxes[k, 0:3], gt_boxes[k, 3:6], gt_boxes[k, 6]

        # relative coordinates
        obj_points[:, 0:3] -= obj_center
        obj_points = common_utils.rotate_points_along_z(np.expand_dims(obj_points, axis=0), -ry).squeeze(0)
        new_lwh = lwh * scale_noises[k][try_idx]

        obj_points[:, 0:3] = obj_points[:, 0:3] * scale_noises[k][try_idx]
        obj_points = common_utils.rotate_points_along_z(np.expand_dims(obj_points, axis=0), ry).squeeze(0)
        # calculate new object center to avoid object float over the road
        obj_center[2] += (new_lwh[2] - lwh[2]) / 2
        obj_points[:, 0:3] += obj_center
        points[point_masks > 0] = obj_points
        gt_boxes[k, 3:6] = new_lwh

        # if enlarge boxes, remove bg points
        if scale_noises[k][try_idx] > 1:
            points_dst_mask = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, 0:3],
                                                                        np.expand_dims(gt_boxes[k],
                                                                                       axis=0)).squeeze(0)

            keep_mask = ~np.logical_xor(point_masks, points_dst_mask)
            points = points[keep_mask]

    return points, gt_boxes


def normalize_object_size(boxes, points, boxes_mask, size_res):
    """
    :param boxes: (N, 7) under unified boxes
    :param points: (N, 3 + C)
    :param boxes_mask
    :param size_res: (3) [l, w, h]
    :return:
    """
    points = copy.deepcopy(points)
    boxes = copy.deepcopy(boxes)
    for k in range(boxes.shape[0]):
        # skip boxes that not need to normalize
        if boxes_mask[k] == 0:
            continue
        masks = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, 0:3], boxes[k:k+1]).squeeze(0)
        obj_points = points[masks > 0]
        obj_center, lwh, ry = boxes[k, 0:3], boxes[k, 3:6], boxes[k, 6]
        obj_points[:, 0:3] -= obj_center
        obj_points = common_utils.rotate_points_along_z(np.expand_dims(obj_points, axis=0), -ry).squeeze(0)
        new_lwh = lwh + np.array(size_res)
        # skip boxes that shift to have negative
        if (new_lwh < 0).any():
            boxes_mask[k] = False
            continue
        scale_lwh = new_lwh / lwh

        obj_points[:, 0:3] = obj_points[:, 0:3] * scale_lwh
        obj_points = common_utils.rotate_points_along_z(np.expand_dims(obj_points, axis=0), ry).squeeze(0)
        # calculate new object center to avoid object float over the road
        obj_center[2] += size_res[2] / 2

        obj_points[:, 0:3] += obj_center
        points[masks > 0] = obj_points
        boxes[k, 3:6] = new_lwh

        # if enlarge boxes, remove bg points
        if (np.array(size_res) > 0).any():
            points_dst_mask = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, 0:3],
                                                                        np.expand_dims(boxes[k],
                                                                                       axis=0)).squeeze(0)

            keep_mask = ~np.logical_xor(masks, points_dst_mask)
            points = points[keep_mask]

    return points, boxes


def rotate_objects(gt_boxes, points, gt_boxes_mask, rotation_perturb, prob, num_try=50):
    """

    Args:
        gt_boxes: [N, 7] (x, y, z, dx, dy, dz, heading) on unified coordinate
        points: [M]
        gt_boxes_mask: [N] bool
        rotation_perturb: ratation noise parameter
        prob: prob to random rotate object
        num_try: times to try rotate one object
    Returns:

    """
    num_boxes = gt_boxes.shape[0]
    if not isinstance(rotation_perturb, (list, tuple, np.ndarray)):
        rotation_perturb = [-rotation_perturb, rotation_perturb]

    # with prob to rotate each object
    rot_mask = np.random.uniform(0, 1, size=[num_boxes]) < prob

    # generate random ratate noise for each boxes
    rot_noise = np.random.uniform(rotation_perturb[0], rotation_perturb[1], size=[num_boxes, num_try])

    for idx in range(num_boxes):
        # don't need to rotate this object
        if (not rot_mask[idx]) or (not gt_boxes_mask[idx]):
            continue

        # generate rotated boxes num_try times
        rot_box = copy.deepcopy(gt_boxes[idx])
        # [num_try, 7]
        rot_box = rot_box.reshape(1, -1).repeat([num_try], axis=0)
        rot_box[:, 6] += rot_noise[idx]

        # detect conflict
        # [num_try, N-1]
        if num_boxes > 1:
            self_mask = np.ones(num_boxes, dtype=np.bool_)
            self_mask[idx] = False
            iou_matrix = iou3d_nms_utils.boxes_bev_iou_cpu(rot_box, gt_boxes[self_mask])
            ious = np.max(iou_matrix, axis=1)
            no_conflict_mask = (ious == 0)
            # all trys have conflict with other gts
            if no_conflict_mask.sum() == 0:
                continue

            # rotate points and assign new box
            try_idx = no_conflict_mask.nonzero()[0][0]
        else:
            try_idx = 0

        point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, 0:3],
                                                                np.expand_dims(gt_boxes[idx], axis=0)).squeeze(0)

        object_points = points[point_masks > 0]
        object_center = gt_boxes[idx][0:3]
        object_points[:, 0:3] -= object_center

        object_points = common_utils.rotate_points_along_z(object_points[np.newaxis, :, :],
                                                           np.array([rot_noise[idx][try_idx]]))[0]

        object_points[:, 0:3] += object_center
        points[point_masks > 0] = object_points

        # remove bg points that lie the position we want to place object
        points_dst_mask = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, 0:3],
                                                                    np.expand_dims(rot_box[try_idx], axis=0)).squeeze(0)

        keep_mask = ~np.logical_xor(point_masks, points_dst_mask)
        points = points[keep_mask]

        gt_boxes[idx] = rot_box[try_idx]

    return gt_boxes, points

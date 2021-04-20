import numpy as np
from ...utils import box_utils

from ..dataset import DatasetTemplate as Dataset


def transform_annotations_to_kitti_format(annos, map_name_to_kitti=None, info_with_fakelidar=False, **kwargs):
    """
    Args:
        annos:
        map_name_to_kitti: dict, map name to KITTI names (Car, Pedestrian, Cyclist)
        info_with_fakelidar:
    Returns:

    """
    for anno in annos:
        if 'name' not in anno:
            anno['name'] = anno['gt_names']
            anno.pop('gt_names')

        for k in range(anno['name'].shape[0]):
            if anno['name'][k] in map_name_to_kitti:
                anno['name'][k] = map_name_to_kitti[anno['name'][k]]
            else:
                anno['name'][k] = 'Person_sitting'

        if 'boxes_lidar' in anno:
            gt_boxes_lidar = anno['boxes_lidar'].copy()
        elif 'gt_boxes_lidar' in anno:
            gt_boxes_lidar = anno['gt_boxes_lidar'].copy()
        else:
            gt_boxes_lidar = anno['gt_boxes'].copy()

        # filter by fov
        if kwargs.get('is_gt', None) and kwargs.get('GT_FILTER', None):
            if kwargs.get('FOV_FILTER', None):
                gt_boxes_lidar = filter_by_fov(anno, gt_boxes_lidar, kwargs)

        # filter by range
        if kwargs.get('GT_FILTER', None) and kwargs.get('RANGE_FILTER', None):
            point_cloud_range = kwargs['RANGE_FILTER']
            gt_boxes_lidar = filter_by_range(anno, gt_boxes_lidar, point_cloud_range, kwargs['is_gt'])

        if kwargs.get('GT_FILTER', None):
            anno['gt_boxes_lidar'] = gt_boxes_lidar

        anno['bbox'] = np.zeros((len(anno['name']), 4))
        anno['bbox'][:, 2:4] = 50  # [0, 0, 50, 50]
        anno['truncated'] = np.zeros(len(anno['name']))
        anno['occluded'] = np.zeros(len(anno['name']))

        if len(gt_boxes_lidar) > 0:
            if info_with_fakelidar:
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(gt_boxes_lidar)

            gt_boxes_lidar[:, 2] -= gt_boxes_lidar[:, 5] / 2
            anno['location'] = np.zeros((gt_boxes_lidar.shape[0], 3))
            anno['location'][:, 0] = -gt_boxes_lidar[:, 1]  # x = -y_lidar
            anno['location'][:, 1] = -gt_boxes_lidar[:, 2]  # y = -z_lidar
            anno['location'][:, 2] = gt_boxes_lidar[:, 0]  # z = x_lidar
            dxdydz = gt_boxes_lidar[:, 3:6]
            anno['dimensions'] = dxdydz[:, [0, 2, 1]]  # lwh ==> lhw
            anno['rotation_y'] = -gt_boxes_lidar[:, 6] - np.pi / 2.0
            anno['alpha'] = -np.arctan2(-gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0]) + anno['rotation_y']
        else:
            anno['location'] = anno['dimensions'] = np.zeros((0, 3))
            anno['rotation_y'] = anno['alpha'] = np.zeros(0)

    return annos


def filter_by_range(anno, gt_boxes_lidar, point_cloud_range, is_gt):
        mask = box_utils.mask_boxes_outside_range_numpy(
            gt_boxes_lidar, point_cloud_range, min_num_corners=1
        )
        gt_boxes_lidar = gt_boxes_lidar[mask]
        anno['name'] = anno['name'][mask]
        if not is_gt:
            anno['score'] = anno['score'][mask]
            anno['pred_labels'] = anno['pred_labels'][mask]

        return gt_boxes_lidar


def filter_by_fov(anno, gt_boxes_lidar, kwargs):
    fov_gt_flag = Dataset.extract_fov_gt(
        gt_boxes_lidar, kwargs['FOV_DEGREE'], kwargs['FOV_ANGLE']
    )
    gt_boxes_lidar = gt_boxes_lidar[fov_gt_flag]
    anno['name'] = anno['name'][fov_gt_flag]

    return gt_boxes_lidar

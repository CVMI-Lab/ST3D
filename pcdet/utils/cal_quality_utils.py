import copy
import os
import torch
import argparse
import pickle
import glob
from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval

import numpy as np
from pcdet.utils import common_utils, box_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils


class QualityMetric(object):
    def __init__(self, infos=None):
        self.quality_metric = {
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'gt': 0,
            'trans_err': 0,
            'scale_err': 0,
            'orient_err': 0
        }
        self.infos = infos

    def check(self, gt_boxes, frame_id, idx):
        assert self.infos[idx]['point_cloud']['lidar_idx'] == frame_id
        assert (self.infos[idx]['annos']['name'] == 'Car').sum() == gt_boxes.shape[0]

    def update(self, pred_boxes, gt_boxes, iou_thresh=0.7, points=None, frame_id=None, idx=None, batch_dict=None):
        remain_mask = gt_boxes[:, 0] != 0
        gt_boxes = gt_boxes[remain_mask]

        self.check(gt_boxes, frame_id, idx)
        tp_boxes, tp_gt_boxes = self.count_tp_fp_fn_gt(
            pred_boxes, gt_boxes, iou_thresh=iou_thresh, points=points
        )
        if tp_boxes is not None and tp_boxes.shape[0] > 0:
            self.cal_tp_metric(tp_boxes, tp_gt_boxes, points=points)

    def count_tp_fp_fn_gt(self, pred_boxes, gt_boxes, iou_thresh=0.7, points=None):
        """ Count the number of tp, fp, fn and gt. Return tp boxes and their corresponding gt boxes

        """
        assert gt_boxes.shape[1] == 7 and pred_boxes.shape[1] == 7
        # import ipdb; ipdb.set_trace(context=20)
        self.quality_metric['gt'] += gt_boxes.shape[0]

        if gt_boxes.shape[0] == 0:
            self.quality_metric['fp'] += pred_boxes.shape[0]
            return None, None
        elif pred_boxes.shape[0] == 0:
            self.quality_metric['fn'] += gt_boxes.shape[0]
            return None, None

        # import ipdb; ipdb.set_trace(context=20)
        # from pcdet.datasets.dataset import DatasetTemplate
        # DatasetTemplate.__vis__(points, gt_boxes, pred_boxes)

        pred_boxes, _ = common_utils.check_numpy_to_torch(pred_boxes)
        gt_boxes, _ = common_utils.check_numpy_to_torch(gt_boxes)

        if not (pred_boxes.is_cuda and gt_boxes.is_cuda):
            pred_boxes, gt_boxes = pred_boxes.cuda(), gt_boxes.cuda()

        iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes[:, :7], gt_boxes[:, :7])
        max_ious, match_idx = torch.max(iou_matrix, dim=1)
        assert max_ious.shape[0] == pred_boxes.shape[0]

        # max iou > iou_thresh is tp
        tp_mask = max_ious >= iou_thresh
        ntps = tp_mask.sum().item()
        self.quality_metric['tp'] += ntps
        self.quality_metric['fp'] += max_ious.shape[0] - ntps

        # gt boxes that missed by tp boxes are fn boxes
        self.quality_metric['fn'] += gt_boxes.shape[0] - ntps

        # get tp boxes and their corresponding gt boxes
        tp_boxes = pred_boxes[tp_mask]
        tp_gt_boxes = gt_boxes[match_idx[tp_mask]]

        if ntps > 0:
            scale_diff, debug_boxes = self.cal_scale_diff(tp_boxes, tp_gt_boxes)
            self.quality_metric['scale_err'] += scale_diff

        return tp_boxes.cpu().numpy(), tp_gt_boxes.cpu().numpy()

    @staticmethod
    def cal_scale_diff(tp_boxes, gt_boxes):
        assert tp_boxes.shape[0] == gt_boxes.shape[0]

        aligned_tp_boxes = tp_boxes.detach().clone()

        # shift their center together
        aligned_tp_boxes[:, 0:3] = gt_boxes[:, 0:3]

        # align their angle
        aligned_tp_boxes[:, 6] = gt_boxes[:, 6]

        iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(aligned_tp_boxes[:, 0:7], gt_boxes[:, 0:7])

        max_ious, _ = torch.max(iou_matrix, dim=1)

        scale_diff = (1 - max_ious).sum().item()

        return scale_diff, aligned_tp_boxes.cpu().numpy()

    @staticmethod
    def cor_angle_range(angle):
        """ correct angle range to [-pi, pi]

        Args:
            angle:

        Returns:

        """
        gt_pi_mask = angle > np.pi
        lt_minus_pi_mask = angle < - np.pi
        angle[gt_pi_mask] = angle[gt_pi_mask] - 2 * np.pi
        angle[lt_minus_pi_mask] = angle[lt_minus_pi_mask] + 2 * np.pi

        return angle

    def cal_angle_diff(self, angle1, angle2):
        """ angle is from x to y, anti-clockwise

        """
        angle1 = self.cor_angle_range(angle1)
        angle2 = self.cor_angle_range(angle2)

        diff = np.abs(angle1 - angle2)
        gt_pi_mask = diff > np.pi
        diff[gt_pi_mask] = 2 * np.pi - diff[gt_pi_mask]

        return diff

    def cal_tp_metric(self, tp_boxes, gt_boxes, points=None):
        assert tp_boxes.shape[0] == gt_boxes.shape[0]

        # L2 distance xy only
        center_distance = np.linalg.norm(tp_boxes[:, :2] - gt_boxes[:, :2], axis=1)
        self.quality_metric['trans_err'] += center_distance.sum()

        # Angle difference
        angle_diff = self.cal_angle_diff(tp_boxes[:, 6], gt_boxes[:, 6])

        assert angle_diff.sum() >= 0

        self.quality_metric['orient_err'] += angle_diff.sum()

        return

    def statistics_result(self, logger=None):
        self.quality_metric['trans_err'] /= self.quality_metric['tp']
        self.quality_metric['scale_err'] /= self.quality_metric['tp']
        self.quality_metric['orient_err'] /= self.quality_metric['tp']

        result = "=============Quality Metrif of Pseudo labels=============\n"
        for key, value in self.quality_metric.items():
            result += '{} : {:.3f}\n'.format(key, value)

        if logger is not None:
            logger.info(result)
            return
        else:
            return result


class QualityMetricPkl(QualityMetric):
    def update(self, pred_boxes, gt_boxes, iou_thresh=0.7, points=None,
               frame_id=None, idx=None, batch_dict=None):
        tp_boxes, tp_gt_boxes = self.count_tp_fp_fn_gt(
            pred_boxes, gt_boxes, iou_thresh=iou_thresh, points=points
        )
        if tp_boxes is not None and tp_boxes.shape[0] > 0:
            self.cal_tp_metric(tp_boxes, tp_gt_boxes, points=points)


def get_quality_of_single_info(pred_infos, gt_infos, class_name):
    pred_infos = pickle.load(open(pred_infos, 'rb'))
    gt_infos = pickle.load(open(gt_infos, 'rb'))
    gt_annos = [info['annos'] for info in gt_infos]

    ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
        gt_annos, pred_infos, current_classes=['Car']
    )
    print(ap_result_str)

    assert len(pred_infos) == len(gt_annos)
    quality_metric = QualityMetricPkl()
    for pred_info, gt_anno in zip(pred_infos, gt_annos):
        pred_boxes = pred_info['boxes_lidar']
        pred_boxes[:, 2] += pred_boxes[:, 5] / 2

        gt_mask = gt_anno['name'] == class_name

        valid_num = gt_anno['gt_boxes_lidar'].shape[0]
        gt_boxes = gt_anno['gt_boxes_lidar'][gt_mask[:valid_num]]

        assert gt_boxes.shape[0] == gt_mask.sum()
        assert (gt_anno['name'][valid_num:] == 'DontCare').all()
        gt_boxes = box_utils.boxes3d_kitti_fakelidar_to_lidar(gt_boxes)

        quality_metric.update(pred_boxes, gt_boxes)

    result = quality_metric.statistics_result()
    print(result)


def get_error_of_multiple_infos(info_path_list, gt_info_path, class_name, iou_thresh=0.7):
    pred_info_list = [pickle.load(open(cur_path, 'rb')) for cur_path in info_path_list]
    gt_infos = pickle.load(open(gt_info_path, 'rb'))
    gt_annos = [info['annos'] for info in gt_infos]

    num_infos = len(pred_info_list)
    quality_metric_list = [QualityMetricPkl() for _ in range(num_infos)]

    print(f'------Start to estimate the errors by considering multiple infos (iou_thresh={iou_thresh})..------')
    # import pdb
    # pdb.set_trace()
    for k, gt_anno in enumerate(gt_annos):
        gt_mask = gt_anno['name'] == class_name
        valid_num = gt_anno['gt_boxes_lidar'].shape[0]
        gt_boxes = gt_anno['gt_boxes_lidar'][gt_mask[:valid_num]]
        assert gt_boxes.shape[0] == gt_mask.sum()
        assert (gt_anno['name'][valid_num:] == 'DontCare').all()

        gt_boxes = box_utils.boxes3d_kitti_fakelidar_to_lidar(gt_boxes)
        gt_boxes, _ = common_utils.check_numpy_to_torch(gt_boxes)

        if gt_boxes.shape[0] == 0:
            continue

        gt_of_tp_mask = np.ones(gt_boxes.shape[0], dtype=np.int)
        for info_idx in range(num_infos):
            pred_boxes = pred_info_list[info_idx][k]['boxes_lidar']
            pred_boxes[:, 2] += pred_boxes[:, 5] / 2
            if pred_boxes.__len__() == 0:
                gt_of_tp_mask[:] = 0
                break

            pred_boxes, _ = common_utils.check_numpy_to_torch(pred_boxes)

            iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes[:, :7].cuda(), gt_boxes[:, :7].cuda())
            max_iou_of_gt, _ = torch.max(iou_matrix, dim=0)
            max_iou_of_gt = max_iou_of_gt.cpu().numpy()
            gt_of_tp_mask[max_iou_of_gt < iou_thresh] = 0

        intersect_gt_boxes = gt_boxes[gt_of_tp_mask > 0]

        for info_idx in range(num_infos):
            pred_boxes = pred_info_list[info_idx][k]['boxes_lidar']
            quality_metric_list[info_idx].update(pred_boxes, intersect_gt_boxes)

    for info_idx in range(num_infos):
        result = quality_metric_list[info_idx].statistics_result()
        print(f'{result} for file: {info_path_list[info_idx]}')


def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--gt_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--class_name', type=str, nargs='+', default='Car', help='')
    parser.add_argument('--iou_thresh', type=float, default=0.7, help='')
    args = parser.parse_args()
    if os.path.isdir(args.pred_infos):
        info_path_list = glob.glob(os.path.join(args.pred_infos, '*.pkl'))
        get_error_of_multiple_infos(info_path_list, args.gt_infos, args.class_name, iou_thresh=args.iou_thresh)
    else:
        get_quality_of_single_info(args.pred_infos, args.gt_infos, args.class_name)


if __name__ == '__main__':
    main()

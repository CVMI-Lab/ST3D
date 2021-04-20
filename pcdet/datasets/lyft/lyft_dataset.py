import copy
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils, box_utils, self_training_utils
from ..dataset import DatasetTemplate


class LyftDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.infos = []
        self.include_lyft_data(self.mode)

    def include_lyft_data(self, mode):
        self.logger.info('Loading lyft dataset')
        lyft_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                lyft_infos.extend(infos)

        self.infos.extend(lyft_infos)
        self.logger.info('Total samples for lyft dataset: %d' % (len(lyft_infos)))

    @staticmethod
    def remove_ego_points(points, center_radius=1.0):
        mask = ~((np.abs(points[:, 0]) < center_radius*1.5) & (np.abs(points[:, 1]) < center_radius))
        return points[mask]

    def get_lidar(self, index):
        info = self.infos[index]
        lidar_path = self.root_path / info['ref_info']['LIDAR_TOP']['lidar_path']
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1)
        if points.shape[0] % 5 != 0:
            points = points[: points.shape[0] - (points.shape[0] % 5)]
        points = points.reshape([-1, 5])[:, :4]
        points = self.remove_ego_points(points, center_radius=1.5)

        return points

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        points = self.get_lidar(index)

        if self.dataset_cfg.get('SHIFT_COOR', None):
            points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)

        input_dict = {
            'points': points,
            'frame_id': Path(info['ref_info']['LIDAR_TOP']['lidar_path']).stem,
            'metadata': {'token': info['token']}
        }

        if 'gt_boxes' in info:
            input_dict.update({
                'gt_boxes': info['gt_boxes'],
                'gt_names': info['gt_names']
            })

            if self.dataset_cfg.get('SHIFT_COOR', None):
                input_dict['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR

            if self.dataset_cfg.get('USE_PSEUDO_LABEL', None) and self.training:
                input_dict['gt_boxes'] = None

            # for debug only
            # gt_boxes_mask = np.array([n in self.class_names for n in input_dict['gt_names']], dtype=np.bool_)
            # debug_dict = {'gt_boxes': copy.deepcopy(input_dict['gt_boxes'][gt_boxes_mask])}

        if self.dataset_cfg.get('FOV_POINTS_ONLY', None):
            input_dict['points'] = self.extract_fov_data(
                input_dict['points'], self.dataset_cfg.FOV_DEGREE, self.dataset_cfg.FOV_ANGLE
            )
            if input_dict['gt_boxes'] is not None:
                fov_gt_flag = self.extract_fov_gt(
                    input_dict['gt_boxes'], self.dataset_cfg.FOV_DEGREE, self.dataset_cfg.FOV_ANGLE
                )
                input_dict.update({
                    'gt_names': input_dict['gt_names'][fov_gt_flag],
                    'gt_boxes': input_dict['gt_boxes'][fov_gt_flag],
                })

        if self.dataset_cfg.get('USE_PSEUDO_LABEL', None) and self.training:
            self.fill_pseudo_labels(input_dict)

        data_dict = self.prepare_data(data_dict=input_dict)

        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False):
            gt_boxes = data_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            data_dict['gt_boxes'] = gt_boxes

        if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]

        return data_dict

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:
        Returns:
        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7]), 'pred_labels': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            if self.dataset_cfg.get('SHIFT_COOR', None):
                pred_boxes[:, 0:3] -= self.dataset_cfg.SHIFT_COOR

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    def kitti_eval(self, eval_det_annos, eval_gt_annos, class_names):
        from ..kitti.kitti_object_eval_python import eval as kitti_eval

        map_name_to_kitti = {
            'car': 'Car',
            'pedestrian': 'Pedestrian',
            'truck': 'Truck',
        }

        def transform_to_kitti_format(annos, info_with_fakelidar=False, is_gt=False):
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
                else:
                    gt_boxes_lidar = anno['gt_boxes'].copy()

                # filter by range
                if self.dataset_cfg.get('GT_FILTER', None) and \
                        self.dataset_cfg.GT_FILTER.RANGE_FILTER:
                    if self.dataset_cfg.GT_FILTER.get('RANGE', None):
                        point_cloud_range = self.dataset_cfg.GT_FILTER.RANGE
                    else:
                        point_cloud_range = self.point_cloud_range
                        point_cloud_range[2] = -10
                        point_cloud_range[5] = 10

                    mask = box_utils.mask_boxes_outside_range_numpy(gt_boxes_lidar,
                                                                    point_cloud_range,
                                                                    min_num_corners=1)
                    gt_boxes_lidar = gt_boxes_lidar[mask]
                    anno['name'] = anno['name'][mask]
                    if not is_gt:
                        anno['score'] = anno['score'][mask]
                        anno['pred_labels'] = anno['pred_labels'][mask]

                # filter by fov
                if is_gt and self.dataset_cfg.get('GT_FILTER', None):
                    if self.dataset_cfg.GT_FILTER.get('FOV_FILTER', None):
                        fov_gt_flag = self.extract_fov_gt(
                            gt_boxes_lidar, self.dataset_cfg['FOV_DEGREE'], self.dataset_cfg['FOV_ANGLE']
                        )
                        gt_boxes_lidar = gt_boxes_lidar[fov_gt_flag]
                        anno['name'] = anno['name'][fov_gt_flag]

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

        # self.filter_det_results(eval_det_annos, self.oracle_infos)
        transform_to_kitti_format(eval_det_annos)
        transform_to_kitti_format(eval_gt_annos, info_with_fakelidar=False, is_gt=True)

        kitti_class_names = []
        for x in class_names:
            if x in map_name_to_kitti:
                kitti_class_names.append(map_name_to_kitti[x])
            else:
                kitti_class_names.append('Person_sitting')
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
            gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
        )
        return ap_result_str, ap_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        if kwargs['eval_metric'] == 'kitti':
            eval_det_annos = copy.deepcopy(det_annos)
            eval_gt_annos = copy.deepcopy(self.infos)
            return self.kitti_eval(eval_det_annos, eval_gt_annos, class_names)
        else:
            raise NotImplementedError

    def create_groundtruth_database(self, used_classes=None):
        import torch

        database_save_path = self.root_path / f'gt_database_withvelo'
        db_info_save_path = self.root_path / f'lyft_dbinfos_withvelo.pkl'

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        for idx in tqdm(range(len(self.infos))):
            sample_idx = idx
            info = self.infos[idx]
            points = self.get_lidar(idx)
            gt_boxes = info['gt_boxes']
            gt_names = info['gt_names']

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            for i in range(gt_boxes.shape[0]):
                filename = '%s_%s_%d.bin' % (sample_idx, gt_names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or gt_names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': gt_names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    if gt_names[i] in all_db_infos:
                        all_db_infos[gt_names[i]].append(db_info)
                    else:
                        all_db_infos[gt_names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)


def create_lyft_info(version, data_path, save_path, split):
    from lyft_dataset_sdk.lyftdataset import LyftDataset
    from . import lyft_utils
    data_path = data_path / version
    save_path = save_path / version
    split_path = data_path.parent / 'ImageSets'

    if split is not None:
        save_path = save_path / split
        split_path = split_path / split

    save_path.mkdir(exist_ok=True)

    assert version in ['trainval', 'one_scene', 'test']

    if version == 'trainval':
        train_split_path = split_path / 'train.txt'
        val_split_path = split_path / 'val.txt'
    elif version == 'test':
        train_split_path = split_path / 'test.txt'
        val_split_path = None
    elif version == 'one_scene':
        train_split_path = split_path / 'one_scene.txt'
        val_split_path = split_path / 'one_scene.txt'
    else:
        raise NotImplementedError

    train_scenes = [x.strip() for x in open(train_split_path).readlines()] if train_split_path.exists() else []
    val_scenes = [x.strip() for x in open(val_split_path).readlines()] if val_split_path.exists() else []

    lyft = LyftDataset(json_path=data_path / 'data', data_path=data_path, verbose=True)

    available_scenes = lyft_utils.get_available_scenes(lyft)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])
    val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])

    print('%s: train scene(%d), val scene(%d)' % (version, len(train_scenes), len(val_scenes)))

    train_lyft_infos, val_lyft_infos = lyft_utils.fill_trainval_infos(
        data_path=data_path, lyft=lyft, train_scenes=train_scenes, val_scenes=val_scenes,
        test='test' in version
    )

    if version == 'test':
        print('test sample: %d' % len(train_lyft_infos))
        with open(save_path / f'lyft_infos_test.pkl', 'wb') as f:
            pickle.dump(train_lyft_infos, f)
    else:
        print('train sample: %d, val sample: %d' % (len(train_lyft_infos), len(val_lyft_infos)))
        with open(save_path / f'lyft_infos_train.pkl', 'wb') as f:
            pickle.dump(train_lyft_infos, f)
        with open(save_path / f'lyft_infos_val.pkl', 'wb') as f:
            pickle.dump(val_lyft_infos, f)


if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_lyft_infos', help='')
    parser.add_argument('--version', type=str, default='train', help='')
    parser.add_argument('--split', type=str, default=None, help='')
    args = parser.parse_args()

    if args.func == 'create_lyft_infos':
        dataset_cfg = EasyDict(yaml.load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        dataset_cfg.VERSION = args.version
        create_lyft_info(
            version=dataset_cfg.VERSION,
            data_path=ROOT_DIR / 'data' / 'lyft',
            save_path=ROOT_DIR / 'data' / 'lyft',
            split=args.split
        )

        lyft_dataset = LyftDataset(
            dataset_cfg=dataset_cfg, class_names=None,
            root_path=ROOT_DIR / 'data' / 'lyft',
            logger=common_utils.create_logger(), training=True
        )
        # lyft_dataset.create_groundtruth_database()

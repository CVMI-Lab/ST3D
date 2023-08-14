import os
from os.path import join
import copy
import pickle
import yaml

import numpy as np
from skimage import io
from scipy.spatial.transform import Rotation as R

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti, self_training_utils
from ..dataset import DatasetTemplate


class JRDBDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ps_label_dir=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger,
            ps_label_dir=ps_label_dir
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None
        
        self.jrdb_infos = []
        self.include_jrdb_data(self.mode)

    def include_jrdb_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading JRDB dataset')
        jrdb_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                jrdb_infos.extend(infos)

        self.jrdb_infos.extend(jrdb_infos)

        if self.logger is not None:
            self.logger.info('Total samples for JRDB dataset: %d' % (len(self.jrdb_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_image_shape(self, idx):
        img_file = self.root_split_path / 'image_2' / ('%s.jpg' % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx) # lidar to camera

        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file, use_coda=True)

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib, margin=0):
        """
        Args:
            pts_rect:
            img_shape:
            calib:
            margin
        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0 - margin, pts_img[:, 0] < img_shape[1] + margin)
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0 - margin, pts_img[:, 1] < img_shape[0] + margin)
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info
            calib = self.get_calib(sample_idx)

            P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}
            info['calib'] = calib_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = np.argwhere( np.isin([obj.cls_type for obj in obj_list], ['DontCare'], invert=True) ).reshape(-1,)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc = annotations['location'][index]
                dims = annotations['dimensions'][index]
                rots = annotations['rotation_y'][index]
                loc_lidar = calib.rect_to_lidar(loc)
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                loc_lidar[:, 2] += h[:, 0] / 2 # shift coord center up by half box

                # ARTHUR CHECKED
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar # successful copy
                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_lidar(sample_idx)
                    calib = self.get_calib(sample_idx)
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])

                    # fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    # pts_fov = points[fov_flag]
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)
                    for k in range(num_objects):
                        # flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                        flag = box_utils.in_hull(pts_rect, corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        if sample_id_list is not None:
            sample_id_list = sample_id_list[:20]
            with futures.ThreadPoolExecutor(num_workers) as executor:
                infos = executor.map(process_single_scene, sample_id_list)
        else:
            infos = []
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('jrdb_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

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
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index]

            if self.dataset_cfg.get('SHIFT_COOR', None):
                pred_boxes[:, 0:3] -= self.dataset_cfg.SHIFT_COOR

            # BOX FILTER
            if self.dataset_cfg.get('TEST', None) and self.dataset_cfg.TEST.BOX_FILTER['FOV_FILTER']:
                box_preds_lidar_center = pred_boxes[:, 0:3]
                pts_rect = calib.lidar_to_rect(box_preds_lidar_center)
                fov_flag = self.get_fov_flag(pts_rect, image_shape, calib, margin=5)
                pred_boxes = pred_boxes[fov_flag]
                pred_labels = pred_labels[fov_flag]
                pred_scores = pred_scores[fov_flag]
            
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes # ARTHUR CHECKED

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)
            
            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)
    
        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.jrdb_infos[0].keys():
            return None, {}

        from ..kitti.kitti_object_eval_python import eval as kitti_eval
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.jrdb_infos] # KITTI cam loc
        
        gt_file = kwargs['output_path'] / 'gt_annos.pkl'
        with open(gt_file, 'wb') as f:
            pickle.dump(eval_gt_annos, f)
            print("Dumping gt file: ", gt_file)
        dt_file = kwargs['output_path'] / 'dt_annos.pkl'
        with open(dt_file, 'wb') as f:
            pickle.dump(eval_det_annos, f)
            print("Dumping dt file: ", dt_file)

        eval_metric = kwargs['eval_metric']
        if eval_metric=="kitti": #KITTI eval metric doesn't work atm
            # Evaluate with KITTI's metric
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)
        elif eval_metric=="jrdb":
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)
            self.dump_annos_to_jrdb(kwargs['output_path'], eval_gt_annos, eval_det_annos)

        return ap_result_str, ap_dict

    def dump_annos_to_jrdb(self, output_dir, gt_annos, dt_annos):
        """
        Convert to annos file to KITTI txt file format
        type truncated occluded num_points alpha bbox(-1, -1, -1, -1) dimensions location rotation_y conf
        """

        if self.split=='val' or self.split=='train':
            filelist_path = self.root_path / 'training/filelist.txt'
        else:
            filelist_path = self.root_path / 'testing/filelist.txt'

        # Make all seq directories for label files
        seq_list = np.loadtxt(filelist_path, dtype=str, usecols=0)
        seq_set = list(set(seq_list.flatten()))
        
        pred_dir = output_dir / 'jrdb_preds'
        gt_dir = output_dir/ 'jrdb_gt'

        for seq in seq_set:
            seq_dir = gt_dir / seq
            if not os.path.exists(seq_dir):
                print(f'Creating jrdb seq dir {seq_dir}')
                os.makedirs(seq_dir)
            seq_dir = pred_dir / seq
            if not os.path.exists(seq_dir):
                print(f'Creating jrdb seq dir {seq_dir}')
                os.makedirs(seq_dir)

        def save_annos_dict(annos_dict, label_path, is_gt=False):
            num_objects = len(annos_dict['name'])

            # Build np array for gt
            otype = annos_dict['name'].astype(str).reshape(-1, 1)
            truncated = annos_dict['truncated'].astype(str).reshape(-1, 1)
            if is_gt:
                occluded = annos_dict['occluded'].astype(str).reshape(-1, 1)
                num_points = annos_dict['num_points_in_gt'].astype(str).reshape(-1, 1) # TODO copy over from annotation file
                conf    = np.array([1] * num_objects).astype(str).reshape(-1, 1)
            else:
                occluded = np.array([0] * num_objects).astype(str).reshape(-1, 1)       # DC
                num_points = np.array([100] * num_objects).astype(str).reshape(-1, 1)   # DC
                conf    = annos_dict['score'].astype(str).reshape(-1, 1)                 # C
                
            alpha = annos_dict['alpha'].astype(str).reshape(-1, 1)
            bbox = annos_dict['bbox'].astype(str)
            dimensions = annos_dict['dimensions'].astype(str)
            location = annos_dict['location'].astype(str)
            rotation_y = annos_dict['rotation_y'].astype(str).reshape(-1, 1)

            label_np = np.hstack((otype, truncated, occluded, num_points, alpha, bbox, dimensions, location, rotation_y, conf))
            
            np.savetxt(label_path, label_np, fmt="%s", delimiter=' ', newline='\n')
        
        # Dump each index to the correct seq directory
        for idx in range(len(gt_annos)):
            info = copy.deepcopy(self.jrdb_infos[idx])
            sample_idx = info['point_cloud']['lidar_idx']
            label_file = f'{sample_idx}.txt'

            seq_dir = seq_list[idx]
            gt_path = gt_dir / seq_dir / label_file
            pred_path = pred_dir / seq_dir / label_file

            save_annos_dict(gt_annos[idx], gt_path, is_gt=True)
            save_annos_dict(dt_annos[idx], pred_path, is_gt=False)

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.jrdb_infos) * self.total_epochs

        return len(self.jrdb_infos)

    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.jrdb_infos)

        info = copy.deepcopy(self.jrdb_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']

        points = self.get_lidar(sample_idx)
        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        img_shape = info['image']['image_shape']

        # if self.dataset_cfg.FOV_POINTS_ONLY: # TODO fix calibrations to use this
        #     pts_rect = calib.lidar_to_rect(points[:, 0:3])
        #     fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
        #     points = points[fov_flag]
   
        if self.dataset_cfg.get('SHIFT_COOR', None): # roughly the same lidar sensor height as coda
            points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
        
        input_dict = {
            'points': points,
            'frame_id': sample_idx,
            'calib': calib,
            'image_shape': img_shape
        }
        
        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            if self.dataset_cfg.get('SHIFT_COOR', None):
                gt_boxes_lidar[:, 0:3] += self.dataset_cfg.SHIFT_COOR

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

            if "gt_boxes2d" in get_item_list:
                input_dict['gt_boxes2d'] = annos["bbox"]

            if self.dataset_cfg.get('REMOVE_ORIGIN_GTS', None) and self.training:
                input_dict['points'] = box_utils.remove_points_in_boxes3d(input_dict['points'], input_dict['gt_boxes'])
                mask = np.zeros(gt_boxes_lidar.shape[0], dtype=np.bool_)
                input_dict['gt_boxes'] = input_dict['gt_boxes'][mask]
                input_dict['gt_names'] = input_dict['gt_names'][mask]

            if self.dataset_cfg.get('USE_PSEUDO_LABEL', None) and self.training:
                input_dict['gt_boxes'] = None

            # for debug only
            # gt_boxes_mask = np.array([n in self.class_names for n in input_dict['gt_names']], dtype=np.bool_)
            # debug_dict = {'gt_boxes': copy.deepcopy(gt_boxes_lidar[gt_boxes_mask])}

            # road_plane = self.get_road_plane(sample_idx)
            # if road_plane is not None:
            #     input_dict['road_plane'] = road_plane

        # load saved pseudo label for unlabel data
        if self.dataset_cfg.get('USE_PSEUDO_LABEL', None) and self.training:
            self.fill_pseudo_labels(input_dict)

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict


def create_jrdb_infos(dataset_cfg, class_names, data_path, save_path, workers=1):
    dataset = JRDBDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('jrdb_infos_%s.pkl' % train_split)
    val_filename = save_path / ('jrdb_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'jrdb_infos_trainval.pkl'
    test_filename = save_path / 'jrdb_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    jrdb_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(jrdb_infos_train, f)
    print('JRDB info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    jrdb_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(jrdb_infos_val, f)
    print('JRDB info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(jrdb_infos_train + jrdb_infos_val, f)
    print('JRDB info trainval file is saved to %s' % trainval_filename)

    dataset.set_split('test')
    jrdb_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(jrdb_infos_test, f)
    print('JRDB info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')

def save_calib(calib_in_path, calib_out_dir, frame_list):
    # Write each split's calibrations to individual .txt files like in kitti
    cam_path = join(calib_in_path, "cameras.yaml")
    lidar_path = join(calib_in_path, "defaults.yaml")

    cam_file = open(cam_path, "r")
    lidar_file = open(lidar_path, "r")
    cam_calibrations = yaml.safe_load(cam_file)
    lidar_calibrations = yaml.safe_load(lidar_file)
    
    Tr_lidarupper_to_cams = []
    camera_calibs = []
    calib_context = ''
    sensor_list = [0, 1]
    
    WIMG, HIMG = lidar_calibrations['image']['width'], lidar_calibrations['image']['height']
    for sensor_id in sensor_list:
        # Save transform lidar to cameras
        Tr_lidarupper_to_cyl = np.zeros((3, 4))
        # lidar_rotvec = np.array(lidar_calibrations['calibrated']['lidar_upper_to_rgb']['rotation'], dtype=float)
        # Tr_lidarupper_to_cyl[:3, :3] = R.from_rotvec(lidar_rotvec).as_matrix()

        # Tr_lidarupper_to_cyl[:3, 3] = np.array(lidar_calibrations['calibrated']['lidar_upper_to_rgb']['translation'])
        # theta = lidar_calibrations['calibrated']['lidar_upper_to_rgb']['rotation'][-1]
        # rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        # Tr_lidarupper_to_cyl_mat1 = np.eye(4)
        # Tr_lidarupper_to_cyl_mat1[:2, :2]   = rotation_matrix
        # Tr_lidarupper_to_cyl_mat1[:3, 3]    = -np.array(lidar_calibrations['calibrated']['lidar_upper_to_rgb']['translation'])

        # Manual coordinate transform from JRDB LiDAR to KITTI Camera
        # Tr_lidarupper_to_cyl[:3, :3] = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])

        Tr_lidarupper_to_cyl_4X4 = np.eye(4)
        Tr_lidarupper_to_cyl_4X4[:3, :3] = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])

        # Tr_lidarupper_to_cyl[:3, :] = (Tr_lidarupper_to_cyl_4X4 @ Tr_lidarupper_to_cyl_mat1)[:3, :]
        Tr_lidarupper_to_cyl[:3, :3] = Tr_lidarupper_to_cyl_4X4[:3, :3]

        # Convert 3D rot vec to matrix
        R_cam0 = np.fromstring(cam_calibrations['cameras'][f'sensor_{sensor_id}']['R'], dtype=float, sep=' ').reshape(3,3)
        T_cam0 = np.fromstring(cam_calibrations['cameras'][f'sensor_{sensor_id}']['T'], dtype=float, sep=' ').reshape(3,1)
        Tr_cam0_wcs = np.hstack((R_cam0, T_cam0))
        
        K_cam0 = np.fromstring(cam_calibrations['cameras'][f'sensor_{sensor_id}']['K'], dtype=float, sep=' ').reshape(3, 3)
        P_cam0 = K_cam0 @ Tr_cam0_wcs

        # Save projection matrix for cameras
        camera_calib = [f'{i:e}' for i in P_cam0.flatten()]
        camera_calibs.append(camera_calib)

        Tr_lidarupper_to_cams.append([f'{i:e}' for i in Tr_lidarupper_to_cyl.flatten()])
    
    # Save rectification matrix for coplanar just identity (since projection matrix rectifies already)

    # TempR0 = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    R0_rect = [f'{i:e}' for i in np.eye(3).flatten()]

    cam_ids = [0, 1]
    for cam_id in cam_ids:
        calib_context += 'P' + str(cam_id) + ': ' + \
            ' '.join(camera_calibs[cam_id]) + '\n'
    calib_context += 'R0_rect' + ': ' + ' '.join(R0_rect) + '\n'
    for cam_id in cam_ids:
        calib_context += f'Tr_velo_to_cam_{cam_id}: ' + \
            ' '.join(Tr_lidarupper_to_cams[cam_id]) + '\n'

    for frame_idx in frame_list:
        with open(
                f'{calib_out_dir}/' +
                f'{str(frame_idx).zfill(6)}.txt',
                'w+') as fp_calib:
            fp_calib.write(calib_context)
            fp_calib.close()

def convert_jrdb_to_kitti(root_path):
    train_input_set = root_path / 'data/jrdb/training/filelist.txt'
    test_input_set = root_path / 'data/jrdb/testing/filelist.txt'

    trainval_fullset = np.loadtxt(train_input_set, dtype=str, delimiter=" ") # Nx1
    # Split train to train and validation based on JRDB recommendation
    val_loc_list = [
        "clark-center-2019-02-28_1", "gates-ai-lab-2019-02-08_0", "huang-2-2019-01-25_0", "meyer-green-2019-03-16_0",
        "nvidia-aud-2019-04-18_0", "tressider-2019-03-16_1", "tressider-2019-04-26_2"
    ]
    val_mask    = np.isin(trainval_fullset[:, 0], val_loc_list)
    val_out_set = trainval_fullset[val_mask, 1] # Nx1

    train_mask      = np.isin(trainval_fullset[:, 0], val_loc_list, invert=True)
    train_out_set   = trainval_fullset[train_mask, 1] # Nx1

    test_out_set    = np.loadtxt(test_input_set, dtype=str, delimiter=" ", usecols=1) # Nx1

    # Dump to imageset files
    imageset_dir = join(root_path, "data/jrdb/ImageSets")
    if not os.path.exists(imageset_dir):
        print(f'Making ImageSets directory: {imageset_dir}')
        os.mkdir(imageset_dir)

    train_path = join(imageset_dir, "train.txt")
    val_path = join(imageset_dir, "val.txt")
    test_path = join(imageset_dir, "test.txt")

    np.savetxt(train_path, train_out_set, fmt="%s")
    np.savetxt(val_path, val_out_set, fmt="%s")
    np.savetxt(test_path, test_out_set, fmt="%s")

    # Convert calib files to correct format
    in_splits = ["training", "testing"]
    for split in in_splits:
        calib_in_path   = root_path / f'data/jrdb/{split}/calib_jrdb'
        calib_out_path  = root_path / f'data/jrdb/{split}/calib'
        if not os.path.exists(calib_out_path):
            print(f'Making calib directory: {calib_out_path}')
            os.mkdir(calib_out_path)
        
        # Get left and right forward facing cameras 0 and 1 is forward, 0 is top row, 1 is bottom row
        if split=="training":
            save_calib(calib_in_path, calib_out_path, train_out_set)
            save_calib(calib_in_path, calib_out_path, val_out_set)
        else:
            save_calib(calib_in_path, calib_out_path, test_out_set)


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_jrdb_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        convert_jrdb_to_kitti(ROOT_DIR)
        create_jrdb_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Pedestrian'],
            data_path=ROOT_DIR / 'data' / 'jrdb',
            save_path=ROOT_DIR / 'data' / 'jrdb'
        )
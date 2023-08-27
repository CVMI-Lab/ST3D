import argparse
import glob
from pathlib import Path
import time
import copy
import json
import os

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    from visual_utils.open3d_vis_utils import draw_box
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.datasets import CODataset
from pcdet.datasets import JRDBDataset

from pcdet.datasets.coda import coda_utils

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext

        data_file_list = glob.glob(str(root_path / "velodyne" / f'*{self.ext}')) 
        gtbbox_list = glob.glob(str(root_path / "label_all" / f'*.txt'))
        # data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]


        data_file_list.sort()
        gtbbox_list.sort()
        self.sample_file_list = data_file_list
        self.gtbbox_list = gtbbox_list


    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError
        
        gt_labels = np.loadtxt(self.gtbbox_list[index], delimiter=' ', usecols=(0), dtype=str)
        gt_boxes = np.loadtxt(self.gtbbox_list[index], delimiter=' ', usecols=(8,9,10,11,12,13,14))
        
        # original order: h w l x y z yaw
        # desired order: x y z l w h yaw
        # swap x y z (labels) -> z -x -y (unified format))
        # swap l w h (labels) -> w l h (unified format)
        unified_gt_boxes = np.stack((gt_boxes[:, 5], -gt_boxes[:, 3], -gt_boxes[:, 4], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 0], gt_boxes[:, 6]), axis=-1)

        input_dict = {
            'points': points,
            'gt_names': gt_labels,
            'gt_boxes': unified_gt_boxes,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def normalize_color(color):
    normalized_color = [(r / 255, g / 255, b / 255) for r, g, b in color]
    return normalized_color

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    use_dataset = "jrdb"
    gen_video = True
    do_preds = True # Set to true to do inference, otherwise just views ground truth
    show_preds = False
    show_gt = True

    if use_dataset=="coda":
        demo_splits = ["train", "test", "val"] # TODO: use test split later to avoid frame drops
        demo_dataset = CODataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, root_path=Path(args.data_path), logger=logger, use_sorted_imageset=True
        )
        color_map=normalize_color(coda_utils.BBOX_ID_TO_COLOR)
    elif use_dataset=="jrdb":
        demo_dataset = JRDBDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, root_path=Path(args.data_path), logger=logger
        )
        color_map = [(0, 1.0, 0)]
    else:
        demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path(args.data_path), ext=args.ext, logger=logger
        )
    
    model=None
    if do_preds:
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        model.cuda()
        model.eval()

    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    class_names = ['Car', 'Pedestrian', 'Cyclist', 'PickupTruck', 'DeliveryTruck', 'ServiceVehicle', 'UtilityVehicle', 'Scooter', 'Motorcycle', 'FireHydrant', 'FireAlarm', 'ParkingKiosk', 'Mailbox', 'NewspaperDispenser', 'SanitizerDispenser', 'CondimentDispenser', 'ATM', 'VendingMachine', 'DoorSwitch', 'EmergencyAidKit', 'Computer', 'Television', 'Dumpster', 'TrashCan', 'VacuumCleaner', 'Cart', 'Chair', 'Couch', 'Bench', 'Table', 'Bollard', 'ConstructionBarrier', 'Fence', 'Railing', 'Cone', 'Stanchion', 'TrafficLight', 'TrafficSign', 'TrafficArm', 'Canopy', 'BikeRack', 'Pole', 'InformationalSign', 'WallSign', 'Door', 'FloorSign', 'RoomLabel', 'FreestandingPlant', 'Tree', 'Other']

    # pts = open3d.geometry.PointCloud()
    # ref_boxes = None
    # vis = open3d.visualization.Visualizer()
    # vis.create_window()

    # vis.get_render_option().point_size = 2.0
    # vis.get_render_option().background_color = np.zeros(3)

    if gen_video:
        V.visualize_3d(demo_dataset, model, logger, color_map, save_vid_filename="test_split.avi", show_gt=show_gt, show_preds=show_preds)
    else:
        with torch.no_grad():
            for idx, data_dict in enumerate(demo_dataset):
                logger.info(f'Visualized sample index: \t{idx + 1}')
                data_dict = demo_dataset.collate_batch([data_dict])
                load_data_to_gpu(data_dict)

                if not do_preds:
                    V.draw_scenes(points=data_dict['points'][:, 1:], gt_boxes=data_dict['gt_boxes'])
                else:
                    pred_dicts, _ = model.forward(data_dict)

                    #### BEGIN for visualizing evaluation boxes
                    # Load saved predictions from pkl file
                    # import pickle
                    # preds_path = '/home/arthur/AMRL/Benchmarks/unsupda/ST3D/output/da-coda-jrdb_models/centerhead_full/pvrcnn_32_pedonly/coda32codacfgLR0.010000OPTadam_onecycle/eval/epoch_30/val/final_result/data'
                    # preds_file = preds_path + '/dt_annos.pkl'
                    # gt_file = preds_path + '/gt_annos.pkl'

                    # with open(preds_file, 'rb') as f:
                    #     pred_dict = pickle.load(f)

                    # with open(gt_file, 'rb') as f:
                    #     gt_dict = pickle.load(f)

                    # preds_labels = [1] * len(pred_dict[0]['name'])
                    # gt_labels = [1] * len(gt_dict[0]['name'])

                    # # Generate new gt boxes from loc, lwh
                    # calib = demo_dataset.get_calib("000000")
                    # loc = gt_dict[0]['location']
                    # dims = gt_dict[0]['dimensions']
                    # rots = gt_dict[0]['rotation_y']
                    # loc_lidar = calib.rect_to_lidar(loc)

                    # l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]

                    # loc_lidar[:, 2] += h[:, 0] / 2
                    # gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)

                    # # Generate new pred boxes from loc, lwh
                    # loc = pred_dict[0]['location']
                    # dims = pred_dict[0]['dimensions']
                    # rots = pred_dict[0]['rotation_y']
                    # loc_lidar = calib.rect_to_lidar(loc)

                    # l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]

                    # loc_lidar[:, 2] += h[:, 0] / 2
                    # pred_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                    
                    # V.draw_scenes(
                    #     points=data_dict['points'][:, 1:], ref_boxes=pred_boxes_lidar,
                    #     ref_scores=pred_dict[0]['score'], ref_labels=preds_labels,
                    #     gt_boxes=gt_boxes_lidar
                    # )
                    # V.draw_scenes(
                    #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                    #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],
                    #     gt_boxes=gt_dict[0]['gt_boxes_lidar']
                    # )

                    #### END for visualizing evaluation boxes

                    if show_preds:
                        V.draw_scenes(
                            points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                            ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],
                            gt_boxes=data_dict['gt_boxes']
                        )
                    else:
                        # x y z l w h yaw
                        pc_filepath = demo_dataset.sample_file_list[idx]
                        frame = pc_filepath.split('/')[-1].split('.')[0]
                        traj_idx = int(frame[:2])
                        frame_idx = int(frame[2:])
                        # Save into json files to preds directory for img visualization
                        pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
                        pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()
                        json_dict = {
                            "3dbbox": [],
                            "filetype": "json",
                            "frame": str(frame_idx)
                        }
                        obj_dict_template = {
                            "classId": "",
                            "instanceId": "",
                            "cX": 0.0,
                            "cY": 0.0,
                            "cZ": 0.0,
                            "h": 0.0,
                            "l": 0.0,
                            "w": 0.0,
                            "labelAttributes": {
                                "isOccluded": "None"
                            },
                            "r": 0.0,
                            "p": 0.0,
                            "y": 0.0
                        }

                        for box_idx, box in enumerate(pred_boxes):
                            label_idx = pred_labels[box_idx]
                            label_name = class_names[label_idx-1]
                            obj_dict = copy.deepcopy(obj_dict_template)
                            obj_dict["cX"] = float(box[0])
                            obj_dict["cY"] = float(box[1])
                            obj_dict["cZ"] = float(box[2])
                            obj_dict["l"] = float(box[3])
                            obj_dict["w"] = float(box[4])
                            obj_dict["h"] = float(box[5])
                            obj_dict["y"] = float(box[6])
                            obj_dict["classId"] = label_name
                            obj_dict["instanceId"] = "%s:%i" % (label_name, box_idx)
                            json_dict["3dbbox"].append(obj_dict)

                        traj = 7
                        json_filename = "3d_bbox_os1_%i_%i.json" % (traj_idx, frame_idx)
                        traj_dir = os.path.join("preds", str(traj_idx))
                        if not os.path.exists(traj_dir):
                            os.mkdir(traj_dir)
                        json_path = os.path.join(traj_dir, json_filename)
                        json_file = open(json_path, "w+")
                        json_file.write(json.dumps(json_dict, indent=2))
                        json_file.close()

                    # if not OPEN3D_FLAG:
                    #     mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
"""
The Lyft data pre-processing and evaluation is modified from
https://github.com/poodarchu/Det3D
"""

import operator
from functools import reduce
from pathlib import Path

import numpy as np
import tqdm
from pyquaternion import Quaternion
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.geometry_utils import transform_matrix


def get_available_scenes(lyft):
    available_scenes = []
    print('total scene num:', len(lyft.scene))
    for scene in lyft.scene:
        scene_token = scene['token']
        scene_rec = lyft.get('scene', scene_token)
        sample_rec = lyft.get('sample', scene_rec['first_sample_token'])
        sd_rec = lyft.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = lyft.get_sample_data(sd_rec['token'])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break
            # if not sd_rec['next'] == '':
            #     sd_rec = nusc.get('sample_data', sd_rec['next'])
            # else:
            #     has_more_frames = False
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num:', len(available_scenes))
    return available_scenes


def get_sample_data(lyft, sample_data_token):
    sd_rec = lyft.get("sample_data", sample_data_token)
    cs_rec = lyft.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])

    sensor_rec = lyft.get("sensor", cs_rec["sensor_token"])
    pose_rec = lyft.get("ego_pose", sd_rec["ego_pose_token"])

    boxes = lyft.get_boxes(sample_data_token)

    box_list = []
    for box in boxes:
        box.translate(-np.array(pose_rec["translation"]))
        box.rotate(Quaternion(pose_rec["rotation"]).inverse)

        box.translate(-np.array(cs_rec["translation"]))
        box.rotate(Quaternion(cs_rec["rotation"]).inverse)

        box_list.append(box)

    return box_list, pose_rec


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


def fill_trainval_infos(data_path, lyft, train_scenes, val_scenes, test=False):
    train_lyft_infos = []
    val_lyft_infos = []
    progress_bar = tqdm.tqdm(total=len(lyft.sample), desc='create_info', dynamic_ncols=True)

    ref_chans = ["LIDAR_TOP", "LIDAR_FRONT_LEFT", "LIDAR_FRONT_RIGHT"]

    for index, sample in enumerate(lyft.sample):
        progress_bar.update()

        ref_info = {}
        for ref_chan in ref_chans:
            if ref_chan not in sample["data"]:
                continue
            ref_sd_token = sample["data"][ref_chan]
            ref_sd_rec = lyft.get("sample_data", ref_sd_token)
            ref_cs_token = ref_sd_rec["calibrated_sensor_token"]
            ref_cs_rec = lyft.get("calibrated_sensor", ref_cs_token)

            ref_to_car = transform_matrix(
                ref_cs_rec["translation"],
                Quaternion(ref_cs_rec["rotation"]),
                inverse=False,
            )

            ref_from_car = transform_matrix(
                ref_cs_rec["translation"],
                Quaternion(ref_cs_rec["rotation"]),
                inverse=True,
            )

            ref_lidar_path = lyft.get_sample_data_path(ref_sd_token)

            ref_info[ref_chan] = {
                "lidar_path": Path(ref_lidar_path).relative_to(data_path).__str__(),
                "ref_from_car": ref_from_car,
                "ref_to_car": ref_to_car,
            }

            if ref_chan == "LIDAR_TOP":
                ref_boxes, ref_pose_rec = get_sample_data(lyft, ref_sd_token)
                ref_time = 1e-6 * ref_sd_rec["timestamp"]
                car_from_global = transform_matrix(
                    ref_pose_rec["translation"],
                    Quaternion(ref_pose_rec["rotation"]),
                    inverse=True,
                )

        info = {
            "ref_info": ref_info,
            'token': sample['token'],
            'car_from_global': car_from_global,
            'timestamp': ref_time,
        }

        if not test:
            annotations = [
                lyft.get("sample_annotation", token) for token in sample["anns"]
            ]

            locs = np.array([b.center for b in ref_boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in ref_boxes]).reshape(-1, 3)[:, [1, 0, 2]]
            rots = np.array([quaternion_yaw(b.orientation) for b in ref_boxes]).reshape(
                -1, 1
            )
            velocity = np.array([b.velocity for b in ref_boxes]).reshape(-1, 3)
            names = np.array([b.name for b in ref_boxes])
            tokens = np.array([b.token for b in ref_boxes]).reshape(-1, 1)
            gt_boxes = np.concatenate([locs, dims, rots], axis=1)

            assert len(annotations) == len(gt_boxes)

            info["gt_boxes"] = gt_boxes
            info["gt_boxes_velocity"] = velocity
            info["gt_names"] = names
            info["gt_boxes_token"] = tokens

        if sample["scene_token"] in train_scenes:
            train_lyft_infos.append(info)
        else:
            val_lyft_infos.append(info)

    progress_bar.close()
    return train_lyft_infos, val_lyft_infos


def boxes_lidar_to_nusenes(det_info):
    boxes3d = det_info['boxes_lidar']
    scores = det_info['score']
    labels = det_info['pred_labels']

    box_list = []
    for k in range(boxes3d.shape[0]):
        quat = Quaternion(axis=[0, 0, 1], radians=boxes3d[k, 6])
        velocity = (*boxes3d[k, 7:9], 0.0) if boxes3d.shape[1] == 9 else (0.0, 0.0, 0.0)
        box = Box(
            boxes3d[k, :3],
            boxes3d[k, [4, 3, 5]],  # wlh
            quat, label=labels[k], score=scores[k], velocity=velocity,
        )
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(nusc, boxes, sample_token):
    s_record = nusc.get('sample', sample_token)
    sample_data_token = s_record['data']['LIDAR_TOP']

    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(Quaternion(cs_record['rotation']))
        box.translate(np.array(cs_record['translation']))
        # Move box to global coord system
        box.rotate(Quaternion(pose_record['rotation']))
        box.translate(np.array(pose_record['translation']))
        box_list.append(box)
    return box_list


def transform_det_annos_to_nusc_annos(det_annos, nusc):
    nusc_annos = {
        'results': {},
        'meta': None,
    }

    for det in det_annos:
        annos = []
        box_list = boxes_lidar_to_nusenes(det)
        box_list = lidar_nusc_box_to_global(
            nusc=nusc, boxes=box_list, sample_token=det['metadata']['token']
        )

        for k, box in enumerate(box_list):
            name = det['name'][k]
            if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                if name in ['car', 'construction_vehicle', 'bus', 'truck', 'trailer']:
                    attr = 'vehicle.moving'
                elif name in ['bicycle', 'motorcycle']:
                    attr = 'cycle.with_rider'
                else:
                    attr = None
            else:
                if name in ['pedestrian']:
                    attr = 'pedestrian.standing'
                elif name in ['bus']:
                    attr = 'vehicle.stopped'
                else:
                    attr = None
            attr = attr if attr is not None else max(
                cls_attr_dist[name].items(), key=operator.itemgetter(1))[0]
            nusc_anno = {
                'sample_token': det['metadata']['token'],
                'translation': box.center.tolist(),
                'size': box.wlh.tolist(),
                'rotation': box.orientation.elements.tolist(),
                'velocity': box.velocity[:2].tolist(),
                'detection_name': name,
                'detection_score': box.score,
                'attribute_name': attr
            }
            annos.append(nusc_anno)

        nusc_annos['results'].update({det["metadata"]["token"]: annos})

    return nusc_annos


def format_nuscene_results(metrics, class_names, version='default'):
    result = '----------------Nuscene %s results-----------------\n' % version
    for name in class_names:
        threshs = ', '.join(list(metrics['label_aps'][name].keys()))
        ap_list = list(metrics['label_aps'][name].values())

        err_name =', '.join([x.split('_')[0] for x in list(metrics['label_tp_errors'][name].keys())])
        error_list = list(metrics['label_tp_errors'][name].values())

        result += f'***{name} error@{err_name} | AP@{threshs}\n'
        result += ', '.join(['%.2f' % x for x in error_list]) + ' | '
        result += ', '.join(['%.2f' % (x * 100) for x in ap_list])
        result += f" | mean AP: {metrics['mean_dist_aps'][name]}"
        result += '\n'

    result += '--------------average performance-------------\n'
    details = {}
    for key, val in metrics['tp_errors'].items():
        result += '%s:\t %.4f\n' % (key, val)
        details[key] = val

    result += 'mAP:\t %.4f\n' % metrics['mean_ap']
    result += 'NDS:\t %.4f\n' % metrics['nd_score']

    details.update({
        'mAP': metrics['mean_ap'],
        'NDS': metrics['nd_score'],
    })

    return result, details

# Copyright (c) AMRL. All rights reserved.
r"""Adapted from `Waymo to KITTI converter
    <https://github.com/caizhongang/waymo_kitti_converter>`_.
"""

from glob import glob
import os
import json
from os.path import join, isfile
import shutil

import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R

from PIL import Image

import tqdm
from multiprocessing import Pool

class CODa2KITTI(object):
    """CODa to KITTI converter.
    This class serves as the converter to change the CODa raw data to KITTI
    format.
    Args:
        load_dir (str): Directory to load CODa raw data.
        save_dir (str): Directory to save data in KITTI format.
        prefix (str): Prefix of filename. In general, 0 for training, 1 for
            validation and 2 for testing.
        workers (int, optional): Number of workers for the parallel process.
        test_mode (bool, optional): Whether in the test_mode. Default: False.
    """

    def __init__(self,
                 load_dir,
                 save_dir,
                 workers=64,
                 split="training",
                 test_mode=False,
                 channels=128):
        self.filter_empty_3dboxes = True
        self.filter_no_label_zone_points = True

        self.lidar_list = [
            'os1'
        ]
        self.type_list = [
            'CAR', 
            'PEDESTRIAN',
            'BIKE', 
            'MOTORCYCLE',
            'SCOOTER',
            'TREE',
            'TRAFFIC SIGN',
            'CANOPY',
            'TRAFFIC LIGHT',
            'BIKE RACK',
            'BOLLARD',
            'PARKING KIOSK',
            'MAILBOX',
            'FIRE HYDRANT',
            'FREESTANDING PLANT',
            'POLE',
            'INFORMATIONAL SIGN',
            'DOOR',
            'FENCE',
            'RAILING',
            'CONE',
            'CHAIR',
            'BENCH',
            'TABLE',
            'TRASH CAN',
            'NEWSPAPER DISPENSER',
            'ROOM LABEL',
            'STANCHION',
            'SANITIZER DISPENSER',
            'CONDIMENT DISPENSER',
            'VENDING MACHINE',
            'EMERGENCY AID KIT',
            'FIRE EXTINGUISHER',
            'COMPUTER',
            'OTHER',
            'HORSE',
            'PICKUP TRUCK',  
            'DELIVERY TRUCK', 
            'SERVICE VEHICLE', 
            'UTILITY VEHICLE', 
            'FIRE ALARM',
            'ATM',
            'CART',
            'COUCH',
            'TRAFFIC ARM',
            'WALL SIGN',
            'FLOOR SIGN',
            'DOOR SWITCH',
            'EMERGENCY PHONE',
            'DUMPSTER',
            'SEGWAY',
            'BUS',
            'SKATEBOARD',
            'WATER FOUNTAIN'
            # Classes below have not been annotated       
            # 'GOLF CART'
            # 'TRUCK'
            # 'CONSTRUCTION BARRIER'
            # 'TELEVISION',
            # 'VACUUM CLEANER',       
        ]
        self.coda_to_kitti_class_map = {
            # Full Class List
            'CAR': 'Car',
            'PEDESTRIAN': 'Pedestrian',
            'BIKE': 'Cyclist',
            'MOTORCYCLE': 'Motorcycle',
            'SCOOTER': 'Scooter',
            'TREE': 'Tree',
            'TRAFFIC SIGN': 'TrafficSign',
            'CANOPY': 'Canopy',
            'TRAFFIC LIGHT': 'TrafficLight',
            'BIKE RACK': 'BikeRack',
            'BOLLARD': 'Bollard',
            'CONSTRUCTION BARRIER': 'ConstructionBarrier',
            'PARKING KIOSK': 'ParkingKiosk',
            'MAILBOX': 'Mailbox',
            'FIRE HYDRANT': 'FireHydrant',
            'FREESTANDING PLANT': 'FreestandingPlant',
            'POLE': 'Pole',
            'INFORMATIONAL SIGN': 'InformationalSign',
            'DOOR': 'Door',
            'FENCE': 'Fence',
            'RAILING': 'Railing',
            'CONE': 'Cone',
            'CHAIR': 'Chair',
            'BENCH': 'Bench',
            'TABLE': 'Table',
            'TRASH CAN': 'TrashCan',
            'NEWSPAPER DISPENSER': 'NewspaperDispenser',
            'ROOM LABEL': 'RoomLabel',
            'STANCHION': 'Stanchion',
            'SANITIZER DISPENSER': 'SanitizerDispenser',
            'CONDIMENT DISPENSER': 'CondimentDispenser',
            'VENDING MACHINE': 'VendingMachine',
            'EMERGENCY AID KIT': 'EmergencyAidKit',
            'FIRE EXTINGUISHER': 'FireExtinguisher',
            'COMPUTER': 'Computer',
            'TELEVISION': 'Television',
            'OTHER': 'Other',
            'HORSE': 'Other',
            'PICKUP TRUCK': 'PickupTruck',  
            'DELIVERY TRUCK': 'DeliveryTruck', 
            'SERVICE VEHICLE': 'ServiceVehicle', 
            'UTILITY VEHICLE': 'UtilityVehicle',
            'FIRE ALARM': 'FireAlarm',
            'ATM': 'ATM',
            'CART': 'Cart',
            'COUCH': 'Couch',
            'TRAFFIC ARM': 'TrafficArm',
            'WALL SIGN': 'WallSign',
            'FLOOR SIGN': 'FloorSign',
            'DOOR SWITCH': 'DoorSwitch',
            'EMERGENCY PHONE': 'EmergencyPhone',
            'DUMPSTER': 'Dumpster',
            'VACUUM CLEANER': 'VacuumCleaner',
            'SEGWAY': 'Segway',
            'BUS': 'Bus',
            'SKATEBOARD': 'Skateboard',
            'WATER FOUNTAIN': 'WaterFountain'
        }
        #MAP Classes not found in KITTI to DontCare
        for class_type in self.type_list:
            class_name = class_type.upper()
            if class_name not in self.coda_to_kitti_class_map.keys():
                self.coda_to_kitti_class_map[class_name] = 'DontCare'

        self.coda_to_kitti_occlusion = {
            "None":     0,
            "unknown":  0, 
            "Unknown":  0,
            "Light":    1,
            "Medium":   1,
            "Heavy":    2,
            "Full":     2
        }

        self.load_dir = load_dir
        if split=="validation" or split=="training":
            self.save_dir = join(save_dir, "training")
        else:
            self.save_dir = join(save_dir, split)
        self.workers = int(workers)
        self.split = split
        self.test_mode = test_mode

        self.label_save_dir = f'{self.save_dir}/label_'
        self.label_all_save_dir = f'{self.save_dir}/label_all'
        self.image_save_dir = f'{self.save_dir}/image_'
        self.calib_save_dir = f'{self.save_dir}/calib'
        self.point_cloud_save_dir = f'{self.save_dir}/velodyne'
        self.pose_save_dir = f'{self.save_dir}/pose'
        self.timestamp_save_dir = f'{self.save_dir}/timestamp'
        self.imageset_save_dir = f'{save_dir}/ImageSets'

        self.bbox_label_files = []
        self.image_files = [] # Store cam0 as paths
        self.lidar_files = []

        self.cam_ids = [0, 1]
        self.sens_name_to_id = {
            'cam0': 0,
            'cam1': 1,
            'os1': 2
        }

        # Used to downsample lidar vertical channels
        self.channels = channels

        self.process_metadata()
        self.create_folder()
        self.create_imagesets()

    def process_metadata(self):
        metadata_path = join(self.load_dir, "metadata")
        assert os.path.exists(metadata_path), "Metadata directory %s does not exist" % metadata_path

        metadata_files = glob("%s/*.json" % metadata_path)
        metadata_files = sorted(metadata_files, key=lambda fname: int(fname.split('/')[-1].split('.')[0]) )

        for mfile in metadata_files:
            assert os.path.isfile(mfile), '%s does not exist' % mfile
            meta_json = json.load(open(mfile, "r"))

            label_list = meta_json["ObjectTracking"][self.split]
            self.bbox_label_files.extend(label_list)

            lidar_list = [label_path.replace('3d_label', '3d_raw').replace('.json', '.bin') 
                for label_path in label_list]
            self.lidar_files.extend(lidar_list)

            image_list = [label_path.replace('3d_label', '2d_rect')
                .replace('os1', 'cam0').replace('.json', '.png') for label_path in label_list]
            self.image_files.extend(image_list)

    def create_imagesets(self):
        if self.split=="testing":
            imageset_file = "test.txt"
        elif self.split=="training":
            imageset_file = "train.txt"
        elif self.split=="validation":
            imageset_file = "val.txt"

        imageset_path = join(self.imageset_save_dir, imageset_file)
        imageset_fp = open(imageset_path, 'w+')
        
        for lidar_path in self.lidar_files:
            lidar_file = lidar_path.split('/')[-1]
            _, _, traj, frame_idx = self.get_filename_info(lidar_file)
            frame_name = f'{str(traj).zfill(2)}{str(frame_idx).zfill(5)}'
            imageset_fp.write(frame_name+'\n')

        imageset_fp.close()

    def convert(self):
        """Convert action."""
        print('Start converting ...')


        pool = Pool(processes=self.workers)

        file_list = list(range(len(self)))
        for _ in tqdm.tqdm(pool.imap_unordered(self.convert_one, [(self, i) for i in file_list]), total=len(file_list)):
            pass
        print('\nFinished ...')
    
    @staticmethod
    def get_filename_info(filename):
        filename_prefix  = filename.split('.')[0]
        filename_prefix  = filename_prefix.split('_')
        
        modality        = filename_prefix[0]+"_"+filename_prefix[1]
        sensor_name     = filename_prefix[2]
        trajectory      = filename_prefix[3]
        frame           = filename_prefix[4]
        return (modality, sensor_name, trajectory, frame)

    @staticmethod
    def set_filename_by_prefix(modality, sensor_name, trajectory, frame):
        if "2d_rect"==modality:
            filetype = "jpg" # change to jpg later
        elif "2d_bbox"==modality:
            filetype = "txt"
        elif "3d_raw"==modality:
            filetype = "bin"
        elif "3d_bbox"==modality:
            filetype = "json"
        sensor_filename = "%s_%s_%s_%s.%s" % (
            modality, 
            sensor_name, 
            trajectory,
            frame,
            filetype
            )
        return sensor_filename

    @staticmethod
    def get_calibration_info(filepath):
        filename = filepath.split('/')[-1]
        filename_prefix = filename.split('.')[0]
        filename_split = filename_prefix.split('_')

        calibration_info = None
        src, tar = filename_split[1], filename_split[-1]
        if len(filename_split) > 3:
            #Sensor to Sensor transform
            extrinsic = yaml.safe_load(open(filepath, 'r'))
            calibration_info = extrinsic
        else:
            #Intrinsic transform
            intrinsic = yaml.safe_load(open(filepath, 'r'))
            calibration_info = intrinsic
        
        return calibration_info, src, tar

    def load_calibrations(self, outdir, trajectory):
        calibrations_path = os.path.join(outdir, "calibrations", str(trajectory))
        calibration_fps = [os.path.join(calibrations_path, file) for file in os.listdir(calibrations_path) if file.endswith(".yaml")]

        calibrations = {}
        for calibration_fp in calibration_fps:
            cal, src, tar = self.get_calibration_info(calibration_fp)
            cal_id = "%s_%s"%(src, tar)

            if cal_id not in calibrations.keys():
                calibrations[cal_id] = {}

            calibrations[cal_id].update(cal)
        
        return calibrations

    def convert_one(self, args):
        """Convert action for single file.
        Args:
            file_idx (int): Index of the file to be converted.
        """
        _, file_idx = args
        relpath = self.bbox_label_files[file_idx]
        filename = relpath.split('/')[-1]
        fullpath = join(self.load_dir, relpath)
        _, _, traj, frame_idx = self.get_filename_info(filename)
        
        for cam_id in self.cam_ids:
            cam = "cam%i" % cam_id
            img_file = self.set_filename_by_prefix("2d_rect", cam, traj, frame_idx) #change to rect later
            img_path = join(self.load_dir, '2d_rect', cam, str(traj), img_file)

            cam_id = cam[-1]
            self.save_image(traj, img_path, cam_id, frame_idx, file_idx)
            if not self.test_mode:
                self.save_label(traj, cam_id, frame_idx, file_idx)
        
        calibrations_path = os.path.join(self.load_dir, "calibrations", str(traj))
        self.save_calib(traj, frame_idx, file_idx)

        self.save_lidar(traj, frame_idx, file_idx, self.channels)

        self.save_pose(traj, frame_idx, file_idx)
        self.save_timestamp(traj, frame_idx, file_idx)

    def __len__(self):
        """Length of the filename list."""
        return len(self.bbox_label_files)

    def save_image(self, traj, src_img_path, cam_id, frame_idx, file_idx):
        """Parse and save the images in jpg format. Jpg is the original format
        used by Waymo Open dataset. Saving in png format will cause huge (~3x)
        unnesssary storage waste.

        Assumes images are rectified
        Args:
            frame_path (str): Absolute filepath to image file 
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        assert isfile(src_img_path), "Image file does not exist: %s" % src_img_path
        kitti_img_path = f'{self.image_save_dir}{str(cam_id)}/' + \
                f'{str(traj).zfill(2)}{str(frame_idx).zfill(5)}.jpg'
        shutil.copyfile(src_img_path, kitti_img_path)

    def save_calib(self, traj, frame_idx, file_idx):
        """Parse and save the calibration data.
        Args:
            calib_path (str): Filepath to calibration file
            traj (int): Current trajectory index.
        """
        calibrations = self.load_calibrations(self.load_dir, traj) # TODO figure out import
    
        # Save transform lidar to cameras
        Tr_os1_to_cam0 = np.array(calibrations['os1_cam0']['extrinsic_matrix']['data']).reshape(4,4)
        
        R_cam0_to_cam1 = np.array(calibrations['cam0_cam1']['extrinsic_matrix']['R']['data']).reshape(3,3)
        T_cam0_to_cam1 = np.array(calibrations['cam0_cam1']['extrinsic_matrix']['T']).reshape(3, 1)
        Tr_cam0_to_cam1 = np.eye(4)
        Tr_cam0_to_cam1[:3, :] = np.hstack((R_cam0_to_cam1, T_cam0_to_cam1))
        
        Tr_os1_to_cam1 = Tr_cam0_to_cam1 @ Tr_os1_to_cam0
        Tr_os1_to_cams_np = [
            Tr_os1_to_cam0[:3, :].reshape((12, )), Tr_os1_to_cam1[:3, :].reshape((12, ))
        ]

        camera_calibs = []
        Tr_os1_to_cams = []
        calib_context = ''
        for cam_id in self.cam_ids:
            # Save projection matrix for cameras
            cam = "cam%i" % cam_id
            camera_calib = calibrations['%s_intrinsics'%cam]['projection_matrix']['data']
            camera_calib = [f'{i:e}' for i in camera_calib]
            camera_calibs.append(camera_calib)

            Tr_os1_to_cams.append([f'{i:e}' for i in Tr_os1_to_cams_np[cam_id]])

        # Save rectification matrix for coplanar just identity (since images are rectified)
        R0_rect = [f'{i:e}' for i in np.eye(3).flatten()]

        for cam_id in self.cam_ids:
            calib_context += 'P' + str(cam_id) + ': ' + \
                ' '.join(camera_calibs[cam_id]) + '\n'
        calib_context += 'R0_rect' + ': ' + ' '.join(R0_rect) + '\n'
        for cam_id in self.cam_ids:
            calib_context += 'Tr_velo_to_cam_' + str(cam_id) + ': ' + \
                ' '.join(Tr_os1_to_cams[cam_id]) + '\n'

        with open(
                f'{self.calib_save_dir}/' +
                f'{str(traj).zfill(2)}{str(frame_idx).zfill(5)}.txt',
                'w+') as fp_calib:
            fp_calib.write(calib_context)
            fp_calib.close()

    def save_lidar(self, traj, frame_idx, file_idx, channels=128):
        """Parse and save the lidar data in psd format.
        Args:
            traj (int): Current trajectory index.
            frame_idx (int): Current frame index.
        """
        bin_file = self.set_filename_by_prefix("3d_raw", "os1", traj, frame_idx)
        bin_path = join(self.load_dir, "3d_raw", "os1", traj, bin_file)
        assert isfile(bin_path), "Bin file for traj %s frame %s does not exist: %s" % (traj, frame_idx, bin_path)
        point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

        point_cloud = self.downsample_lidar(point_cloud, channels)
    
        pc_path = f'{self.point_cloud_save_dir}/' + \
            f'{str(traj).zfill(2)}{str(frame_idx).zfill(5)}.bin'
 
        point_cloud.astype(np.float32).tofile(pc_path)

    def downsample_lidar(self, pc, channels):
        # Downsamples by selecting vertical channels with different step sizes
        vert_ds = 128 // int(channels)
        pc = pc[:, :4].reshape(128, 1024, -1)
        ds_pc = pc[np.arange(0, 128, vert_ds), :, :]
        return ds_pc.reshape(-1, 4)

    def save_label(self, traj, cam_id, frame_idx, file_idx):
        """Parse and save the label data in txt format.
        The relation between coda and kitti coordinates is noteworthy:
        1. x, y, z correspond to l, w, h (coda) -> l, h, w (kitti)
        2. x-y-z: front-left-up (coda) -> right-down-front(kitti)
        3. bbox origin at volumetric center (coda) -> bottom center (kitti)
        4. rotation: +x around y-axis (kitti) -> +x around z-axis (coda)
        Args:
            traj (str): Current trajectory index.
            frame_idx (str): Current frame index.
        """
        anno_file   = self.set_filename_by_prefix("3d_bbox", "os1", traj, frame_idx)
        anno_path   = join(self.load_dir, "3d_bbox", "os1", traj, anno_file)
        anno_dict   = json.load(open(anno_path))

        twod_anno_file  = self.set_filename_by_prefix("2d_bbox", "cam0", traj, frame_idx)
        twod_anno_path = join(self.load_dir, "2d_bbox", "cam0", traj, twod_anno_file)

        twod_anno_dict = np.loadtxt(twod_anno_path, dtype=int).reshape(-1, 6)

        calibrations = self.load_calibrations(self.load_dir, traj) # TODO figure out import
        # Save transform lidar to cameras
        Tr_os1_to_camx = np.array(calibrations['os1_cam0']['extrinsic_matrix']['data']).reshape(4,4)
        
        if cam_id==1:
            R_cam0_to_cam1 = np.array(calibrations['cam0_cam1']['extrinsic_matrix']['R']['data']).reshape(3,3)
            T_cam0_to_cam1 = np.array(calibrations['cam0_cam1']['extrinsic_matrix']['T']).reshape(3, 1)
            Tr_cam0_to_cam1 = np.eye(4)
            Tr_cam0_to_cam1[:3, :] = np.hstack((R_cam0_to_cam1, T_cam0_to_cam1))
            
            Tr_os1_to_camx = Tr_cam0_to_cam1 @ Tr_os1_to_camx

        fp_label_all = open(
            f'{self.label_all_save_dir}/' +
            f'{str(traj).zfill(2)}{str(frame_idx).zfill(5)}.txt', 'w+')
        id_to_bbox = dict()
        id_to_name = dict()

        #Reset fp_label file if it exists
        fp_label = open(
            f'{self.label_save_dir}{cam_id}/' +
            f'{str(traj).zfill(2)}{str(frame_idx).zfill(5)}.txt', 'w')
        fp_label.close()
        for anno_idx, tred_anno in enumerate(anno_dict["3dbbox"]):
            bounding_box = None
            name = tred_anno['classId'].upper()
            if name not in self.type_list:
                # print("Unknown class %s found, skipping..."%name)
                continue

            class_idx = self.type_list.index(name)

            if name not in self.coda_to_kitti_class_map:
                # print("Class %s not mapped to KITTI, skipping..."%name)
                continue
            my_type = self.coda_to_kitti_class_map[name]

            #TODO: Add in bbox filtering by number of points
            bounding_box = twod_anno_dict[anno_idx, :]
        
            height = tred_anno['h']
            length = tred_anno['l']
            width = tred_anno['w']

            x = tred_anno['cX']
            y = tred_anno['cY']
            z = tred_anno['cZ'] - height / 2

            # not available
            truncated = 0
            alpha = -10
            coda_occluded = tred_anno['labelAttributes']['isOccluded']
            occluded = self.coda_to_kitti_occlusion[coda_occluded]

            pt_ref = Tr_os1_to_camx @ \
                np.array([x, y, z, 1]).reshape((4, 1))
            x, y, z, _ = pt_ref.flatten().tolist()

            rotation_y = -tred_anno['y'] - np.pi / 2 # Convert to correct direction and limit range
            track_id = tred_anno['instanceId']

            line = my_type + \
                ' {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
                    round(truncated, 2), occluded, round(alpha, 2),
                    round(bounding_box[0], 2), round(bounding_box[1], 2),
                    round(bounding_box[2], 2), round(bounding_box[3], 2),
                    round(height, 2), round(width, 2), round(length, 2),
                    round(x, 2), round(y, 2), round(z, 2),
                    round(rotation_y, 2))

            line_all = line[:-1] + '\n'

            fp_label = open(
                f'{self.label_save_dir}{cam_id}/' +
                f'{str(traj).zfill(2)}{str(frame_idx).zfill(5)}.txt', 'a')
            fp_label.write(line)
            fp_label.close()

            fp_label_all.write(line_all)

        fp_label_all.close()

    def save_pose(self, traj, frame_idx, file_idx):
        """Parse and save the pose data.
        Note that SDC's own pose is not included in the regular training
        of KITTI dataset. KITTI raw dataset contains ego motion files
        but are not often used. Pose is important for algorithms that
        take advantage of the temporal information.
        Args:
            traj (str): Current trajectory.
            frame_idx (str): Current frame index.
        """
        pose_dir = join(self.load_dir, "poses", "dense")
        pose_path = join(pose_dir, "%s.txt"%traj)
        assert isfile(pose_path), "Pose file for traj %s frame %s does not exist: %s" % (traj, frame_idx, pose_path)
        pose_np = np.loadtxt(pose_path, skiprows=int(frame_idx), max_rows=1)

        pose_T = pose_np[1:4].reshape(3, -1)
        pose_quat_xyzw = np.append(pose_np[5:8], pose_np[4])
        pose_R = R.from_quat(pose_quat_xyzw).as_matrix()

        pose_kitti = np.eye(4)
        pose_kitti[:3, :] = np.hstack((pose_R, pose_T))
       
        np.savetxt(
            join(f'{self.pose_save_dir}/' +
                 f'{str(traj).zfill(2)}{str(frame_idx).zfill(5)}.txt'),
            pose_kitti)

    def save_timestamp(self, traj, frame_idx, file_idx):
        """Save the timestamp data in a separate file instead of the
        pointcloud.
        Note that SDC's own pose is not included in the regular training
        of KITTI dataset. KITTI raw dataset contains ego motion files
        but are not often used. Pose is important for algorithms that
        take advantage of the temporal information.
        Args:
            traj (str): Current trajectory
            frame_idx (str): Current frame index.
        """
        ts_dir = join(self.load_dir, "timestamps")
        frame_to_ts_file = "%s.txt" % traj
        frame_to_ts_path = join(ts_dir, frame_to_ts_file)
        ts_s_np = np.loadtxt(frame_to_ts_path, skiprows=int(frame_idx), max_rows=1)
        ts_us_np = int(ts_s_np * 1e6)

        with open(
                join(f'{self.timestamp_save_dir}/' +
                     f'{str(traj).zfill(2)}{str(frame_idx).zfill(5)}.txt'),
                'w') as f:
            f.write(str(ts_us_np))

    def create_folder(self):
        """Create folder for data preprocessing."""
        if not self.test_mode:
            dir_list1 = [
                self.label_all_save_dir, self.calib_save_dir,
                self.point_cloud_save_dir, self.pose_save_dir,
                self.timestamp_save_dir, self.imageset_save_dir
            ]
            dir_list2 = [self.label_save_dir, self.image_save_dir]
        else:
            dir_list1 = [
                self.calib_save_dir, self.point_cloud_save_dir,
                self.pose_save_dir, self.timestamp_save_dir, self.imageset_save_dir
            ]
            dir_list2 = [self.image_save_dir]
        
        for d in dir_list1:
            os.makedirs(d, exist_ok=True)
        for d in dir_list2:
            for i in range(2):
                os.makedirs(f'{d}{str(i)}', exist_ok=True)

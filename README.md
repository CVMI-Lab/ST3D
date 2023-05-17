# ST3D & ST3D++

Code release for the paper **ST3D: Self-training for Unsupervised Domain Adaptation on 3D Object Detection**, CVPR 2021 and
**ST3D++: Denoised Self-training for Unsupervised Domain Adaptation on 3D Object Detection**, T-PAMI 2022.


## News
[2023-05-17] Support Waymo -> Lyft setting.

[2022-09-26] ST3D++ (The extension of ST3D) has been integrated in this repo for Waymo->KITTI and nuScenes->KITTI.


## Introduction
Our code is based on OpenPCDet v0.3.0.
More updates on OpenPCDet are supposed to be compatible with our code.

## Model Zoo

### Waymo -> KITTI TASK

|                                                                                             |     method     | Car@R40 | Ped@R40 | Cyc@R40 | 
|---------------------------------------------------------------------------------------------|:--------------:|:-------:|:-------:|:-------:|
| [SECOND-IoU](tools/cfgs/da-waymo-kitti_models/secondiou_st3d/secondiou_st3d_car.yaml)       |      ST3D      |  62.19  |  48.33  |  46.09  | 
| [SECOND-IoU](tools/cfgs/da-waymo-kitti_models/secondiou_st3d/secondiou_st3d_car.yaml)       |  ST3D (w/ sn)  |  73.62  |  51.92  |  53.00  |
| [SECOND-IoU](tools/cfgs/da-waymo-kitti_models/secondiou_st3d/secondiou_st3d++_ros_car.yaml) |     ST3D++     |  65.10  |  53.87  |  53.43  |
| [SECOND-IoU](tools/cfgs/da-waymo-kitti_models/secondiou_st3d/secondiou_st3d++_sn_car.yaml)  | ST3D++ (w/ sn) |  74.73  |  59.21  |  60.76  |
| [PVRCNN](tools/cfgs/da-waymo-kitti_models/pvrcnn_st3d/pvrcnn_st3d.yaml)                     |      ST3D      |  64.05  |    -    |    -    |
| [PVRCNN](tools/cfgs/da-waymo-kitti_models/pvrcnn_st3d/pvrcnn_st3d.yaml)                     |  ST3D (w/ sn)  |  77.33  |    -    |    -    |

We could not provide the above pretrained models due to [Waymo Dataset License Agreement](https://waymo.com/open/terms/), 
but you should achieve similar performance by training with the default configs. To access these pretrained models, please 
send us an email with your name, institute, a screenshot of the Waymo dataset registration confirmation mail, and your 
intended usage. Please send a second email if we don't get back to you in two days. Please note that Waymo open dataset is 
under strict non-commercial license, so we are not allowed to share the model with you if it will use for any profit-oriented activities.

Also, the training Waymo data used in our work is version 1.0, but the version now available is version 1.2. 
The pretrained model on these two version data should be similar when adapted to KITTI.  


### nuScenes -> KITTI TASK
|                                                                                                |     method     | Car@R40 | Ped@R40 | Cyc@R40 | 
|------------------------------------------------------------------------------------------------|:--------------:|:-------:|:-------:|:-------:|
| [SECOND-IoU](tools/cfgs/da-nuscenes-kitti_models/secondiou/secondiou_old_anchor.yaml)          |  Source Only   |  17.92  |    -    |    -    | 
| [SECOND-IoU](tools/cfgs/da-nuscenes-kitti_models/secondiou/secondiou_old_anchor_ros.yaml)      |      ROS       |  25.37  |    -    |    -    | 
| [SECOND-IoU](tools/cfgs/da-nuscenes-kitti_models/secondiou/secondiou_old_anchor_sn.yaml)       |       SN       |  21.23  |  34.36  |  5.67   |
| [SECOND-IoU](tools/cfgs/da-nuscenes-kitti_models/secondiou_st3d/secondiou_st3d_car.yaml)       |      ST3D      |  55.19  |    -    |    -    |
| [SECOND-IoU](tools/cfgs/da-nuscenes-kitti_models/secondiou_st3d/secondiou_st3d_car.yaml)       |  ST3D (w/ SN)  |  62.27  |    -    |    -    |
| [SECOND-IoU](tools/cfgs/da-nuscenes-kitti_models/secondiou_st3d/secondiou_st3d++_ros_car.yaml) |     ST3D++     |  66.01  |  45.23  |  25.98  |
| [SECOND-IoU](tools/cfgs/da-nuscenes-kitti_models/secondiou_st3d/secondiou_st3d++_sn_car.yaml)  | ST3D++ (w/ SN) |  66.24  |  46.75  |  22.66  |
| [PV-RCNN](tools/cfgs/da-nuscenes-kitti_models/pvrcnn/pvrcnn_old_anchor.yaml)                   |  Source Only   |  37.17  |    -    |    -    |
| [PV-RCNN](tools/cfgs/da-nuscenes-kitti_models/pvrcnn/pvrcnn_old_anchor_ros.yaml)               |      ROS       |  38.84  |    -    |    -    | 
| [PV-RCNN](tools/cfgs/da-nuscenes-kitti_models/pvrcnn/pvrcnn_old_anchor_sn.yaml)                |       SN       |  49.47  |    -    |    -    |
| [PV-RCNN](tools/cfgs/da-nuscenes-kitti_models/pvrcnn_st3d/pvrcnn_st3d.yaml)                    |      ST3D      |  71.11  |    -    |    -    | 
| [PV-RCNN](tools/cfgs/da-nuscenes-kitti_models/pvrcnn_st3d/pvrcnn_st3d.yaml)                    |  ST3D (w/ SN)  |  73.16  |    -    |    -    |
| [PointRCNN](tools/cfgs/da-nuscenes-kitti_models/pointrcnn/pointrcnn.yaml)                      |      ROS       |  55.92  |    -    |    -    |
| [PointRCNN](tools/cfgs/da-nuscenes-kitti_models/pointrcnn_st3d/pointrcnn_st3d++_car.yaml)      |    ST3D++      |  67.51  |    -    |    -    |

We provide pretrained models here for nuScenes > KITTI task in [models](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/jhyang13_connect_hku_hk/ErVtcVax3OBJgn4TyQxbOwMBCt1kDCt4_rYaXqHPsg_ZNw?e=a42lgy).


### Waymo -> nuScenes TASK
|                                             | method | Car@R11 | Car@R40 | download | 
|---------------------------------------------|----------:|:-------:|:-------:|:---------:|
| [SECOND-IoU](tools/cfgs/da-waymo-nus_models/secondiou_st3d/secondiou_st3d.yaml) | ST3D | 23.24 | 20.19 | [model](https://drive.google.com/file/d/150gvNE29iZ8Ah4L4uO_E40x0fyr_cn_J/view?usp=sharing) | 
| [PVRCNN](tools/cfgs/da-waymo-nus_models/pvrcnn_st3d/pvrcnn_st3d.yaml)    | ST3D | 27.18 | 22.99 | [model](https://drive.google.com/file/d/1Gk5i4qRz5tLU7Yd4U-oPureN9cV6TaLa/view?usp=sharing) |
| [SECOND-IoU](tools/cfgs/da-waymo-nus_models/secondiou_st3d/secondiou_st3d.yaml) | ST3D (w/ sn) | 23.52 | 20.38 | [model](https://drive.google.com/file/d/1WUHBTZylVTFLdSvO8wrgti27SEjFJxQl/view?usp=sharing) | 
| [PVRCNN](tools/cfgs/da-waymo-nus_models/pvrcnn_st3d/pvrcnn_st3d.yaml)    | ST3D (w/ sn)   | 28.06 | 23.67 | [model](https://drive.google.com/file/d/1KTCvmltfwIEh9i2t3Pu9APp_2GKyENme/view?usp=sharing) |

We could not provide the above pretrained models due to [Waymo Dataset License Agreement](https://waymo.com/open/terms/), 
but you should achieve similar performance by training with the default configs.


### Waymo -> Lyft TASK
|                                                                                  |       method | Car@R11 | Car@R40 |                                            download                                            | 
|----------------------------------------------------------------------------------|-------------:|:-------:|:-------:|:----------------------------------------------------------------------------------------------:|
| [SECOND-IoU](tools/cfgs/da-waymo-lyft_models/secondiou/secondiou.yaml)           |  Source Only |    -    |  54.34  |                                               -                                                | 
| [SECOND-IoU](tools/cfgs/da-waymo-lyft_models/secondiou/secondiou_sn.yaml)        |           SN |    -    |  54.34  |                                               -                                                |
| [SECOND-IoU](tools/cfgs/da-waymo-lyft_models/secondiou_st3d/secondiou_st3d.yaml) |         ST3D |    -    |  59.24  | [model](https://drive.google.com/file/d/1YfOawQUvdVmExQ5H_-xQuFBCFg57dD01/view?usp=share_link) | 
| [SECOND-IoU](tools/cfgs/da-waymo-lyft_models/secondiou_st3d/secondiou_st3d.yaml) | ST3D (w/ sn) |    -    |  57.99  | [model](https://drive.google.com/file/d/1bZQaDeIxcUzk6fyMI7RRykl-eAnpySCp/view?usp=share_link) | 
| [PVRCNN](tools/cfgs/da-waymo-lyft_models/pvrcnn/pvrcnn.yaml)                     |  Source Only |    -    |  58.53  |                                               -                                                |
| [PVRCNN](tools/cfgs/da-waymo-lyft_models/pvrcnn/pvrcnn_sn.yaml)                  |           SN |    -    |  56.64  |                                               -                                                |
| [PVRCNN](tools/cfgs/da-waymo-lyft_models/pvrcnn_st3d/pvrcnn_st3d.yaml)           |         ST3D |    -    |  60.53  | [model](https://drive.google.com/file/d/12hVyBwT3rT3bm6yWsa5xrBim89iPVkev/view?usp=share_link) |
| [PVRCNN](tools/cfgs/da-waymo-lyft_models/pvrcnn_st3d/pvrcnn_st3d.yaml)           | ST3D (w/ sn) |    -    |  58.54  | [model](https://drive.google.com/file/d/1fYbD_43vGQYaCY767L78-2Znyber5tdf/view?usp=share_link) |

We could not provide the above pretrained models due to [Waymo Dataset License Agreement](https://waymo.com/open/terms/), 
but you should achieve similar performance by training with the default configs.



## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for the installation.

## Getting Started

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to learn more usage about this project.

### Supported features and ToDo List

- [x] Support inference and pre-trained model 

- [x] Support training code on Waymo -> KITTI task

- [x] Update to OpenPCDet v0.3.0 version.
  
- [x] Support more adaptation tasks.

## License

Our code is released under the Apache 2.0 license.

## Acknowledgement

Our code is heavily based on [OpenPCDet v0.3](https://github.com/open-mmlab/OpenPCDet/commit/e3bec15f1052b4827d942398f20f2db1cb681c01). Thanks OpenPCDet Development Team for their awesome codebase.

## Citation

If you find this project useful in your research, please consider cite:
```
@inproceedings{yang2021st3d,
    title={ST3D: Self-training for Unsupervised Domain Adaptation on 3D Object Detection},
    author={Yang, Jihan and Shi, Shaoshuai and Wang, Zhe and Li, Hongsheng and Qi, Xiaojuan},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2021}
}
```
```
@article{yang2021st3d++,
  title={ST3D++: Denoised Self-training for Unsupervised Domain Adaptation on 3D Object Detection},
  author={Yang, Jihan and Shi, Shaoshuai and Wang, Zhe and Li, Hongsheng and Qi, Xiaojuan},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022}
}
```
```
@misc{openpcdet2020,
    title={OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds},
    author={OpenPCDet Development Team},
    howpublished = {\url{https://github.com/open-mmlab/OpenPCDet}},
    year={2020}
}
```

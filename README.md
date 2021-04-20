# ST3D

Code release for the paper **ST3D: Self-training for Unsupervised Domain Adaptation on 3D Object Detection**, CVPR 2021

![framework](./docs/framework.png)

**Authors**: Jihan Yang\*, Shaoshuai Shi\*,  Zhe Wang, Hongsheng Li, Xiaojuan Qi (\*equal contribution)

[[arXiv]](https://arxiv.org/abs/2103.05346);


## Introduction

Our code is based on [OpenPCDet v0.2](https://github.com/open-mmlab/OpenPCDet/tree/v0.2.0). 
More updates on OpenPCDet are supposed to be compatible with our code.

## Model Zoo

### Waymo -> KITTI TASK

|                                             | method | Car@R11 | Car@R40 | download | 
|---------------------------------------------|----------:|:-------:|:-------:|:---------:|
| [SECOND-IoU](tools/cfgs/da-waymo-kitti_models/secondiou_st3d/secondiou_st3d.yaml) | ST3D (w/ sn) | 73.33 | 73.62 | [model](https://drive.google.com/file/d/12f5EwWz2AugC0Ba3NETQG87GqE2C12Xz/view?usp=sharing) | 
| [PVRCNN](tools/cfgs/da-waymo-kitti_models/pvrcnn_st3d/pvrcnn_st3d.yaml)    | ST3D (w/ sn)   | 75.71 | 77.33 | [model](https://drive.google.com/file/d/12f5EwWz2AugC0Ba3NETQG87GqE2C12Xz/view?usp=sharing) |

We could not provide the above pretrained models due to [Waymo Dataset License Agreement](https://waymo.com/open/terms/), 
but you should achieve similar performance by training with the default configs.

Also, the training Waymo data used in our work is version 1.0, but the version now available is version 1.2. 
The pretrained model on these two version data should be similar when adapted to KITTI.  

## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for the installation.

## Getting Started

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to learn more usage about this project.

### Supported features and ToDo List

- [x] Support inference and pre-trained model 

- [x] Support training code on Waymo -> KITTI task

- [ ] Update to Latest OpenPCDet version.
  
- [ ] Support more adaptation tasks.

## License

Our code is released under the Apache 2.0 license.

## Acknowledgement

Our code is heavily based on [OpenPCDet v0.2](https://github.com/open-mmlab/OpenPCDet/tree/v0.2.0). Thanks OpenPCDet Development Team for their awesome codebase.

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
@misc{openpcdet2020,
    title={OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds},
    author={OpenPCDet Development Team},
    howpublished = {\url{https://github.com/open-mmlab/OpenPCDet}},
    year={2020}
}
```

# UT CODa Model Training Configurations
We utilize the ST3D++ code repository and OpenPCDet model training configuration files for all empirical analyses conducted for the UT CODa paper.

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

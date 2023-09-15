# UT Campus Object Dataset (CODa) Object Detection Models

<b>Official model development kit for CODa.</b> We strongly recommend using this repository to run our pretrained
models and train on custom datasets. Thanks to the authors of ST3D++ and OpenPCDet from whom this repository
was adapted from.

![Sequence 0 Clip](./docs/codademo.gif)

## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for the installation.

## Getting Started

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to learn more about how to use this project.

## License

Our code is released under the Apache 2.0 license.

## Paper Citation

If you find our work useful in your research, please consider citing our work:
```
@inproceedings{zhang2023utcoda,
    title={Towards Robust 3D Robot Perception in Urban Environments: The UT Campus Object Dataset},
    author={Arthur Zhang and Chaitanya Eranki and Christina Zhang and Raymond Hong and Pranav Kalyani and Lochana Kalyanaraman and Arsh Gamare and Maria Esteva and Joydeep Biswas },
    booktitle={},
    year={2023}
}
```

## Dataset Citation
```
@data{T8/BBOQMV_2023,
author = {Zhang, Arthur and Eranki, Chaitanya and Zhang, Christina and Hong, Raymond and Kalyani, Pranav and Kalyanaraman, Lochana and Gamare, Arsh and Bagad, Arnav and Esteva, Maria and Biswas, Joydeep},
publisher = {Texas Data Repository},
title = {{UT Campus Object Dataset (CODa)}},
year = {2023},
version = {DRAFT VERSION},
doi = {10.18738/T8/BBOQMV},
url = {https://doi.org/10.18738/T8/BBOQMV}
}
```

## Acknowledgement

Our code is heavily based on [OpenPCDet v0.3](https://github.com/open-mmlab/OpenPCDet/commit/e3bec15f1052b4827d942398f20f2db1cb681c01). Thanks OpenPCDet Development Team for their awesome codebase.


Thank you to the authors of ST3D++ or OpenPCDet for an awesome codebase!
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

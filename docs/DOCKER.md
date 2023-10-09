# Docker Setup

This tutorial walks through how to pull our prebuilt docker image, and how to use it to deploy our pretrained 3D object detection models. Our 3D object detection module currently runs at approximately 5 Hz on an NVIDIA RTX 3090 GPU.

## Setup Docker Container

1. Pulling docker image from Docker Hub. This command will pull the latest prebuilt image from Docker Hub.
```
docker image pull docker://artzha/coda-models
```

2. Start the docker container. This command will run the docker container with the following settings:
    - `--gpus all`: use all GPUs available on the host machine
    - `--net host`: use the host machine's network stack
    - `--rm`: remove the container after exiting
    - `-it`: run the container in interactive mode
```
docker run --gpus all --net host --rm -it artzha/coda-models
```

3. Run inference. This command will publish predicted 3D bounding boxes for point clouds published over ROS on the `YOUR_POINT_CLOUD_TOPIC` topic. Uses 32 channel LiDAR by default.
```
python tools/ros_demo.py --pc {YOUR_POINT_CLOUD_TOPIC}
```

4. (Optional). Using script with custom LiDAR resolutions. Configure the command below to set the sensor resolution used by the container during inference.

***Mount pretrained model weight directory***
```
docker run --gpus all --net host -v {MODEL_WEIGHT_DIRECTORY_PATH}:/coda-models/ckpts --rm -it artzha/coda-models
```

***Run inference using custom model weights***
Replace the `LIDAR_RESOLUTION` placeholder with the desired LiDAR resolution (16, 32, 64, 128). The following command will publish predicted 3D bounding boxes for point clouds published over ROS on the `YOUR_POINT_CLOUD_TOPIC` topic.

```
python tools/ros_demo.py --pc {YOUR_POINT_CLOUD_TOPIC} --cfg_file tools/cfgs/da-coda-coda_models/waymocenterhead/pvrcnn_allclass{LIDAR_RESOLUTION}full_finetune_headfull.yaml --ckpt ckpts/coda{LIDAR_RESOLUTION}_allclass_bestoracle.pth
```
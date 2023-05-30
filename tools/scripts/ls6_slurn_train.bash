#!/bin/bash
#SBATCH -J train_st3d_model               # Job name
#SBATCH -o train_st3d_model.%j            # Name of stdout output file (%j expands to jobId)
#SBATCH -p gpu-a100                        # Queue name
#SBATCH -N 1                               # Total number of nodes requested (128 cores/node)
#SBATCH -n 1                               # Total number of mpi tasks requested
#SBATCH -t 48:00:00                        # Run time (hh:mm:ss)
#SBATCH -A IRI23004                        # Allocation name

export APPTAINERENV_CUDA_VISIBLE_DEVICES=0,1,2
module load cuda/11.3
module load tacc-apptainer
cd /scratch/09156/arthurz/research/ST3D/tools

export NUM_GPUS=3
export CUDA_VISIBLE_DEVICES=0,1,2
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=0,1,2

# For wandb
export WANDB_API_KEY=dfd81f8955f7587d12b13da5256e56f80a89c014

# singularity exec --nv st3d.sif bash scripts/dist_train.sh 3 --cfg_file cfgs/nuscenes_models/pvrcnn/pvrcnn_oracle.yaml --batch_size 3

#For nuscenes oracle
# export PORT1=29500
# export CONFIG_FILE1=cfgs/nuscenes_models/pvrcnn/pvrcnn_oracle.yaml
# export EXTRA_TAG1=pvrcnn_oracle

#For nuscenes -> coda da
# export PORT=29501
# export CONFIG_FILE1=cfgs/da-nuscenes-coda_models/pv_rcnn/pvrcnn_old_anchor.yaml
# export EXTRA_TAG1=pvrcnn_pretrain_coda_small 

# For CODa oracle
# export PORT=29500
# export CONFIG_FILE1=cfgs/coda_models/cbgs_pp_multihead.yaml
# export EXTRA_TAG1=coda_md_oracle_3class

# For CODa 128 allclass to nuscenes head
# export PORT=29500
# export CONFIG_FILE1=cfgs/da-coda-nuscenes_models/pvrcnn_128oracle_finetune_head.yaml
# export EXTRA_TAG1=coda128-md-nus-finetune-head
# export PRETRAINED_MODEL1=../output/coda_models/pvrcnn_oracle_allclass/coda_md_allclass_oracleLR0.010000OPTadam_onecycle/ckpt/checkpoint_epoch_50.pth
# export CKPT1=../output/da-coda-nuscenes_models/pvrcnn_128oracle_finetune_head/coda128-md-nus-finetune-headLR0.010000OPTadam_onecycle/ckpt/checkpoint_epoch_13.pth

# For CODa 128 allclass to nuscenes headfull
export PORT=29500
export CONFIG_FILE1=cfgs/da-coda-nuscenes_models/pvrcnn_128oracle_finetune_headfull.yaml
export EXTRA_TAG1=coda128-md-nus-finetune-headfull
export PRETRAINED_MODEL1=../output/da-coda-nuscenes_models/pvrcnn_128oracle_finetune_head/coda128-md-nus-finetune-headLR0.010000OPTadam_onecycle/ckpt/checkpoint_epoch_12.pth
export CKPT1=../output/da-coda-nuscenes_models/pvrcnn_128oracle_finetune_headfull/coda128-md-nus-finetune-headfullLR0.010000OPTadam_onecycle/ckpt/checkpoint_epoch_16.pth

# For CODa 32 allclass to nuscenes head
# export PORT=29500
# export CONFIG_FILE1=cfgs/da-coda-nuscenes_models/pvrcnn_32oracle_finetune_head.yaml
# export EXTRA_TAG1=coda32-md-nus-finetune-head
# export PRETRAINED_MODEL1=../output/coda_models/pvrcnn_oracle_allclass32/coda32_md_allclass_oracleLR0.010000OPTadam_onecycle/ckpt/checkpoint_epoch_50.pth
# export CKPT1=../output/da-coda-nuscenes_models/pvrcnn_32oracle_finetune_head/coda32-md-nus-finetune-headLR0.010000OPTadam_onecycle/ckpt/checkpoint_epoch_8.pth

# For Waymo to CODa 128 allclass head
# export PORT=29500
# export CONFIG_FILE1=cfgs/da-waymo-coda_models/pvrcnn/pvrcnn_old_anchor_allclass_finetune_head.yaml
# export EXTRA_TAG1=waymo-coda-md-allclass-finetune-head
# export PRETRAINED_MODEL1=../output/da-waymo-coda_models/pvrcnn/pvrcnn_old_anchor/waymo_oracleLR0.010000OPTadam_onecycle/ckpt/checkpoint_epoch_50.pth

# For Waymo to CODa 32 allclass head
# export PORT=29500
# export CONFIG_FILE1=cfgs/da-waymo-coda_models/pvrcnn/pvrcnn_old_anchor_allclass32_finetune_head.yaml
# export EXTRA_TAG1=waymo-coda32-md-allclass-finetune-head
# export PRETRAINED_MODEL1=../output/da-waymo-coda_models/pvrcnn/pvrcnn_old_anchor/waymo_oracleLR0.010000OPTadam_onecycle/ckpt/checkpoint_epoch_50.pth

# For Waymo to CODa x 3class headfull
# export PORT=29500
# export CONFIG_FILE1=cfgs/coda_models/pvrcnn_oracle_3class128.yaml
# export EXTRA_TAG1=coda-md-3class-128-finetune-headfull
# export PRETRAINED_MODEL1=../output/da-waymo-coda_models/pvrcnn/pvrcnn_old_anchor/waymo_oracleLR0.010000OPTadam_onecycle/ckpt/checkpoint_epoch_50.pth

# export PORT=29500
# export CONFIG_FILE1=cfgs/coda_models/pvrcnn_oracle_3class64.yaml
# export EXTRA_TAG1=coda-md-3class-64-finetune-headfull
# export PRETRAINED_MODEL1=../output/da-waymo-coda_models/pvrcnn/pvrcnn_old_anchor/waymo_oracleLR0.010000OPTadam_onecycle/ckpt/checkpoint_epoch_50.pth

# export PORT=29500
# export CONFIG_FILE1=cfgs/coda_models/pvrcnn_oracle_3class32.yaml
# export EXTRA_TAG1=coda-md-3class-32-finetune-headfull
# export PRETRAINED_MODEL1=../output/da-waymo-coda_models/pvrcnn/pvrcnn_old_anchor/waymo_oracleLR0.010000OPTadam_onecycle/ckpt/checkpoint_epoch_50.pth

export PORT=29500
export CONFIG_FILE1=cfgs/coda_models/pvrcnn_oracle_3class16.yaml
export EXTRA_TAG1=coda-md-3class-16-finetune-headfull
export PRETRAINED_MODEL1=../output/da-waymo-coda_models/pvrcnn/pvrcnn_old_anchor/waymo_oracleLR0.010000OPTadam_onecycle/ckpt/checkpoint_epoch_50.pth

# For waymo->coda (SN).
# export PORT=29501
# export CONFIG_FILE1=cfgs/da-waymo-coda_models/pvrcnn/pvrcnn_old_anchor_sn.yaml
# export EXTRA_TAG1=pvrcnn_sn_coda
# export CKPT1=../output/da-waymo-coda_models/pvrcnn/pvrcnn_old_anchor_ros_sn/pvrcnn_sn_coda_small/ckpt/checkpoint_epoch_6.pth
# export EPOCH1=50
# export BATCH_SIZE1=12

# For waymo->coda (SOURCE)
# export PORT=29503
# export CONFIG_FILE1=cfgs/da-waymo-coda_models/pvrcnn/pvrcnn_old_anchor.yaml
# export EXTRA_TAG1=waymo_oracle
# # export CKPT1=../output/da-waymo-coda_models/pvrcnn/pvrcnn_old_anchor/waymo_coda_pretrain/ckpt/checkpoint_epoch_6.pth
# export BATCH_SIZE1=12
# export EPOCH1=30

# Train kitti pp oracle
# export PORT=29500
# export CONFIG_FILE1=cfgs/da-kitti-coda_models/cbgs_pp_multihead.yaml
# export EXTRA_TAG1=kitti_oracle

# Train kitti pvrcnn oracle
# export PORT=29501
# export CONFIG_FILE2=cfgs/da-kitti-coda_models/pvrcnn/pvrcnn_old_anchor.yaml 
# export EXTRA_TAG2=kitti_oracle

# Launch regular models from scratch
# ibrun -n 1 -o 0 task_affinity singularity exec --nv ../st3d_latest.sif bash scripts/dist_train.sh 3 --cfg_file ${CONFIG_FILE1} --extra_tag ${EXTRA_TAG1} >> launcher_train_models_task3
# ibrun -n 1 -o 0 task_affinity singularity exec --nv ../st3d_latest.sif bash scripts/dist_train.sh 3 --cfg_file ${CONFIG_FILE2} --extra_tag ${EXTRA_TAG2} >> launcher_train_models_task2

#Launch pretrained model
ibrun -n 1 -o 0 task_affinity singularity exec --nv ../st3d_latest.sif bash scripts/dist_train.sh 3 --cfg_file ${CONFIG_FILE1} --extra_tag ${EXTRA_TAG1} --pretrained_model ${PRETRAINED_MODEL1} >> launcher_train_models_task3

#Launch model from ckpt
# ibrun -n 1 -o 0 task_affinity singularity exec --nv ../st3d_latest.sif bash scripts/dist_train.sh 3 --cfg_file ${CONFIG_FILE1} --extra_tag ${EXTRA_TAG1} --ckpt ${CKPT1} >> launcher_train_models_task4

# Uncomment to launch model training
# module load launcher_gpu
# export LAUNCHER_WORKDIR=/scratch/09156/arthurz/research/ST3D/tools
# export LAUNCHER_JOB_FILE=scripts/launcher_train_damodels

# Uncomment to launch model training from ckpt
# module load launcher_gpu
# export LAUNCHER_WORKDIR=/scratch/09156/arthurz/research/ST3D/tools
# export LAUNCHER_JOB_FILE=scripts/launcher_train_damodels_ckpt

# ${LAUNCHER_DIR}/paramrun

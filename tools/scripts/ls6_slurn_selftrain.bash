#!/bin/bash
#SBATCH -J selftrain_st3d_model            # Job name
#SBATCH -o selftrain_st3d_model.%j         # Name of stdout output file (%j expands to jobId)
#SBATCH -p gpu-a100                        # Queue name
#SBATCH -N 1                               # Total number of nodes requested (128 cores/node)
#SBATCH -n 3                               # Total number of mpi tasks requested
#SBATCH -t 00:20:00                        # Run time (hh:mm:ss)
#SBATCH -A IRI23004                        # Allocation name

export APPTAINERENV_CUDA_VISIBLE_DEVICES=0,1,2
module load cuda/11.3
module load tacc-apptainer
cd /scratch/09156/arthurz/research/ST3D/tools

export NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=0

# singularity exec --nv st3d.sif bash scripts/dist_train.sh 3 --cfg_file cfgs/nuscenes_models/pvrcnn/pvrcnn_oracle.yaml --batch_size 3

#For nuscenes oracle
# export PORT1=29500
# export CONFIG_FILE1=cfgs/nuscenes_models/pvrcnn/pvrcnn_oracle.yaml
# export EXTRA_TAG1=pvrcnn_oracle

#For nuscenes -> coda da
# export PORT=29501
# export CONFIG_FILE1=cfgs/da-nuscenes-coda_models/pv_rcnn/pvrcnn_old_anchor.yaml
# export EXTRA_TAG1=pvrcnn_pretrain_coda_small 

# For nuScenes -> CODA Self Train
# export PORT=29502
# export CONFIG_FILE1=cfgs/da-nuscenes-coda_models/pvrcnn_st3d/pvrcnn_st3d.yaml
# export EXTRA_TAG1=pvrcnn_oracle_coda_small_resamp 
# export PRETRAINED_MODEL1=../output/da-nuscenes-coda_models/pv_rcnn/pvrcnn_old_anchor/pvrcnn_pretrain_coda_small/ckpt/checkpoint_epoch_49.pth
# export BATCH_SIZE1=8
# export EPOCH1=50

# For waymo->coda da
export PORT=29502
export CONFIG_FILE1=cfgs/da-waymo-coda_models/pvrcnn_st3d/pvrcnn_st3d.yaml
export EXTRA_TAG1=pvrcnn_old_anchor 
export PRETRAINED_MODEL1=../output/da-waymo-coda_models/pv_rcnn/pvrcnn_old_anchor/pvrcnn_pretrain_coda_small/ckpt/checkpoint_epoch_24.pth
export BATCH_SIZE1=8
export EPOCH1=50

# Uncomment to launch model training
module load launcher_gpu
export LAUNCHER_WORKDIR=/scratch/09156/arthurz/research/ST3D/tools
export LAUNCHER_JOB_FILE=scripts/launcher_selftrain_damodels

${LAUNCHER_DIR}/paramrun

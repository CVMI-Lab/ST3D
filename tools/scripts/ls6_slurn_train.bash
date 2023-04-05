#!/bin/bash
#SBATCH -J train_st3d_model               # Job name
#SBATCH -o train_st3d_model.%j            # Name of stdout output file (%j expands to jobId)
#SBATCH -p gpu-a100                        # Queue name
#SBATCH -N 1                               # Total number of nodes requested (128 cores/node)
#SBATCH -n 3                               # Total number of mpi tasks requested
#SBATCH -t 48:00:00                        # Run time (hh:mm:ss)
#SBATCH -A IRI23004                        # Allocation name

export APPTAINERENV_CUDA_VISIBLE_DEVICES=0,1,2
module load cuda/11.3
module load tacc-apptainer
cd /scratch/09156/arthurz/research/ST3D/tools

export NUM_GPUS=3
export CUDA_VISIBLE_DEVICES=0,1,2

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
# export PORT=29502
# export CONFIG_FILE1=cfgs/coda_models/pvrcnn_oracle.yaml
# export EXTRA_TAG1=pvrcnn_oracle_coda_small_resamp 
# export CKPT1=../output/coda_models/pvrcnn_oracle/pvrcnn_oracle_coda_small_resamp/ckpt/checkpoint_epoch_10.pth
# export EPOCH1=80
# export BATCH_SIZE1=30

# For waymo->coda da
export PORT=29503
export CONFIG_FILE1=cfgs/da-waymo-coda_models/pvrcnn/pvrcnn_old_anchor.yaml
export EXTRA_TAG1=waymo_da_coda_small
export CKPT1=../output/da-waymo-coda_models/pvrcnn/pvrcnn_old_anchor/waymo_coda_pretrain/ckpt/checkpoint_epoch_6.pth
export BATCH_SIZE1=12

# Uncomment to launch model training
module load launcher_gpu
export LAUNCHER_WORKDIR=/scratch/09156/arthurz/research/ST3D/tools
export LAUNCHER_JOB_FILE=scripts/launcher_train_damodels

# Uncomment to launch model training from ckpt
# module load launcher_gpu
# export LAUNCHER_WORKDIR=/scratch/09156/arthurz/research/ST3D/tools
# export LAUNCHER_JOB_FILE=scripts/launcher_train_damodels_ckpt

${LAUNCHER_DIR}/paramrun

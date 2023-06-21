#!/bin/bash
#SBATCH -J test_st3d_model               # Job name
#SBATCH -o test_st3d_model.%j            # Name of stdout output file (%j expands to jobId)
#SBATCH -p gpu-a100                        # Queue name
#SBATCH -N 1                               # Total number of nodes requested (128 cores/node)
#SBATCH -n 1                               # Total number of mpi tasks requested
#SBATCH -t 12:00:00                        # Run time (hh:mm:ss)
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

#For nuscenes
# export PORT1=29500
# export CONFIG_FILE1=cfgs/nuscenes_models/pvrcnn/pvrcnn_oracle.yaml
# export EXTRA_TAG1=pvrcnn_oracle

# For nuscenes-coda (source) eval.
# export PORT=29501
# export CONFIG_FILE1=cfgs/da-nuscenes-coda_models/pv_rcnn/pvrcnn_old_anchor.yaml
# export EXTRA_TAG1=pvrcnn_pretrain_coda_small
# export BATCH_SIZE1=36

# For waymo->coda (SOURCE).
# export PORT=29501
# export CONFIG_FILE1=cfgs/da-waymo-coda_models/pvrcnn/pvrcnn_old_anchor.yaml
# export EXTRA_TAG1=waymo_da_coda_small
# export BATCH_SIZE1=36
# export EVAL_TAG1=waymo_da_oracle

# For waymo->waymo (SOURCE).
# export PORT=29501
# export CONFIG_FILE1=cfgs/da-waymo-coda_models/pvrcnn/pvrcnn_old_anchor_oracle.yaml
# export EXTRA_TAG1=waymo_da_coda_small
# export BATCH_SIZE1=12
# export EVAL_TAG1=waymo_da_oracle

# For CODa to CODa evaluation

#CODa16
# export PORT=29500
# export CONFIG_FILE1=cfgs/coda_models/pvrcnn_oracle_3class16.yaml
# export EXTRA_TAG1=coda-md-3class-16-finetune-headfull

# #CODa32
# export PORT=29500
# export CONFIG_FILE1=cfgs/coda_models/pvrcnn_oracle_3class32.yaml
# export EXTRA_TAG1=coda-md-3class-32-finetune-headfull

# #CODa64
export PORT=29500
export CONFIG_FILE1=cfgs/coda_models/pvrcnn_oracle_3class64.yaml
export EXTRA_TAG1=coda-md-3class-64-finetune-headfull

# #CODa128
# export PORT=29500
# export CONFIG_FILE1=cfgs/coda_models/pvrcnn_oracle_3class128.yaml
# export EXTRA_TAG1=coda-md-3class-128-finetune-headfull

#Launch pretrained model
ibrun -n 1 -o 0 task_affinity singularity exec --nv ../st3d_latest.sif bash scripts/dist_test.sh 3 --cfg_file ${CONFIG_FILE1} --extra_tag ${EXTRA_TAG1} --eval_all >> launcher_eval_models_task0

# Uncomment to launch model training
# module load launcher_gpu
# export LAUNCHER_WORKDIR=/scratch/09156/arthurz/research/ST3D/tools
# export LAUNCHER_JOB_FILE=scripts/launcher_test_damodels_source

# ${LAUNCHER_DIR}/paramrun

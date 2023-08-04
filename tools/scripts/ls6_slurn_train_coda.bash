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

# CODa 16 FT head
export PORT=29500
export CONFIG_FILE1=cfgs/da-coda-coda_models/waymocenterhead/pvrcnn_allclass16full_finetune_head.yaml
export EXTRA_TAG1=coda16bestwaymo
export PRETRAINED_MODEL1=../output/da-waymo-coda_models/pvrcnn_codacfg/pvrcnn_pretrain_waymo/bestcodaLR0.010000OPTadam_onecycle/ckpt/checkpoint_epoch_21.pth

# CODa 32 FT head
export PORT=29500
export CONFIG_FILE1=cfgs/da-coda-coda_models/waymocenterhead/pvrcnn_allclass32full_finetune_head.yaml
export EXTRA_TAG1=coda32bestwaymo
export PRETRAINED_MODEL1=../output/da-waymo-coda_models/pvrcnn_codacfg/pvrcnn_pretrain_waymo/bestcodaLR0.010000OPTadam_onecycle/ckpt/checkpoint_epoch_21.pth

# CODa 64 FT head
export PORT=29500
export CONFIG_FILE1=cfgs/da-coda-coda_models/waymocenterhead/pvrcnn_allclass64full_finetune_head.yaml
export EXTRA_TAG1=coda64bestwaymo
export PRETRAINED_MODEL1=../output/da-waymo-coda_models/pvrcnn_codacfg/pvrcnn_pretrain_waymo/bestcodaLR0.010000OPTadam_onecycle/ckpt/checkpoint_epoch_21.pth

# CODa 128 FT head
export PORT=29500
export CONFIG_FILE1=cfgs/da-coda-coda_models/waymocenterhead/pvrcnn_allclass128full_finetune_head.yaml
export EXTRA_TAG1=coda128bestwaymo
export PRETRAINED_MODEL1=../output/da-waymo-coda_models/pvrcnn_codacfg/pvrcnn_pretrain_waymo/bestcodaLR0.010000OPTadam_onecycle/ckpt/checkpoint_epoch_21.pth

# CODa 16 FT headfull
export PORT=29500
export CONFIG_FILE1=cfgs/da-coda-coda_models/waymocenterhead/pvrcnn_allclass16full_finetune_headfull.yaml
export EXTRA_TAG1=coda16bestwaymo
export PRETRAINED_MODEL1=../output/da-coda-coda_models/waymocenterhead/pvrcnn_allclass16full_finetune_head/coda16bestwaymoLR0.010000OPTadam_onecycle/ckpt/checkpoint_epoch_10.pth

# CODa 32 FT headfull
export PORT=29500
export CONFIG_FILE1=cfgs/da-coda-coda_models/waymocenterhead/pvrcnn_allclass32full_finetune_headfull.yaml
export EXTRA_TAG1=coda32bestwaymo
export PRETRAINED_MODEL1=../output/da-coda-coda_models/waymocenterhead/pvrcnn_allclass32full_finetune_head/coda32bestwaymoLR0.010000OPTadam_onecycle/ckpt/checkpoint_epoch_10.pth

# CODa 64 FT headfull
export PORT=29500
export CONFIG_FILE1=cfgs/da-coda-coda_models/waymocenterhead/pvrcnn_allclass64full_finetune_headfull.yaml
export EXTRA_TAG1=coda64bestwaymo
export PRETRAINED_MODEL1=../output/da-coda-coda_models/waymocenterhead/pvrcnn_allclass64full_finetune_head/coda64bestwaymoLR0.010000OPTadam_onecycle/ckpt/checkpoint_epoch_10.pth

# CODa 128 FT headfull
export PORT=29500
export CONFIG_FILE1=cfgs/da-coda-coda_models/waymocenterhead/pvrcnn_allclass128full_finetune_headfull.yaml
export EXTRA_TAG1=coda128bestwaymo
export PRETRAINED_MODEL1=../output/da-coda-coda_models/waymocenterhead/pvrcnn_allclass128full_finetune_head/coda128bestwaymoLR0.010000OPTadam_onecycle/ckpt/checkpoint_epoch_10.pth


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

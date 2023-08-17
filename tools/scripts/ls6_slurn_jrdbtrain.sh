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

# Launch Dataset to JRDB pretraining models

# CODa32 ped only
export PORT=29500
export CONFIG_FILE1=cfgs/da-coda-jrdb_models/centerhead_full/pvrcnn_32_pedonly.yaml
export EXTRA_TAG1=coda32codacfg

# Waymo ped only
export PORT=29500
export CONFIG_FILE1=cfgs/da-waymo-jrdb_models/centerhead_full/pvrcnn_pedonly.yaml
export EXTRA_TAG1=waymocodacfg

# NUS ped only
export PORT=29500
export CONFIG_FILE1=cfgs/da-nus-jrdb_models/centerhead_full/pvrcnn_pedonly.yaml
export EXTRA_TAG1=nuscodacfg

# JRDB scratch ped only no points filter
export PORT=29500
export CONFIG_FILE1=cfgs/jrdb_models/centerhead/pvrcnn_pedonly.yaml
export EXTRA_TAG1=jrdbcodacfgnofilter

# JRDB scratch ped only 25m range (DOES LR SEARCH BY DEFAULT)
export PORT=29500
export CONFIG_FILE1=cfgs/jrdb_models/centerhead/pvrcnn_pedonly.yaml
export EXTRA_TAG1=jrdbcodacfg25mrange

# JRDB scratch ped only full range (DOES LR SEARCH BY DEFAULT)
export PORT=29500
export CONFIG_FILE1=cfgs/jrdb_models/centerhead/pvrcnn_pedonly_fullrange.yaml
export EXTRA_TAG1=jrdbcodacfgfullrange

# JRDB scratch ped only full range all augs (DOES LR SEARCH BY DEFAULT)
export PORT=29500
export CONFIG_FILE1=cfgs/jrdb_models/centerhead/pvrcnn_pedonly_fullrange_allaugs.yaml
export EXTRA_TAG1=jrdbcodacfgfullrange

# JRDB scratch ped only 25m range all augs (DOES LR SEARCH BY DEFAULT)
export PORT=29500
export CONFIG_FILE1=cfgs/jrdb_models/centerhead/pvrcnn_pedonly_allaugs.yaml
export EXTRA_TAG1=jrdbcodacfg25mrange

# JRDB finetune from CODa32
export PORT=29500
export CONFIG_FILE1=cfgs/da-coda-jrdb_models/centerhead_full/pvrcnn_32_pedonly_finetune_headfull.yaml
export EXTRA_TAG1=jrdbcoda32pretrained
export PRETRAINED_MODEL1=../output/da-coda-jrdb_models/centerhead_full/pvrcnn_32_pedonly/coda32codacfgLR0.010000OPTadam_onecycle/ckpt/checkpoint_epoch_2.pth

# Launch regular models from scratch
ibrun -n 1 -o 0 task_affinity singularity exec --nv ../st3d_latest.sif bash scripts/dist_train.sh 3 --cfg_file ${CONFIG_FILE1} --extra_tag ${EXTRA_TAG1} >> launcher_train_models_task0
# ibrun -n 1 -o 0 task_affinity singularity exec --nv ../st3d_latest.sif bash scripts/dist_train.sh 3 --cfg_file ${CONFIG_FILE2} --extra_tag ${EXTRA_TAG2} >> launcher_train_models_task1

#Launch pretrained model
# ibrun -n 1 -o 0 task_affinity singularity exec --nv ../st3d_latest.sif bash scripts/dist_train.sh 3 --cfg_file ${CONFIG_FILE1} --extra_tag ${EXTRA_TAG1} --pretrained_model ${PRETRAINED_MODEL1} >> launcher_train_models_task3

#Launch model from ckpt
# ibrun -n 1 -o 0 task_affinity singularity exec --nv ../st3d_latest.sif bash scripts/dist_train.sh 3 --cfg_file ${CONFIG_FILE1} --extra_tag ${EXTRA_TAG1} --ckpt ${CKPT1} >> launcher_train_models_task4
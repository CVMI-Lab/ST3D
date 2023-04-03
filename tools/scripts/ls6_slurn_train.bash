#!/bin/bash
#SBATCH -J train_st3d_model               # Job name
#SBATCH -o train_st3d_model.%j            # Name of stdout output file (%j expands to jobId)
#SBATCH -p gpu-a100                        # Queue name
#SBATCH -N 1                               # Total number of nodes requested (128 cores/node)
#SBATCH -n 1                               # Total number of mpi tasks requested
#SBATCH -t 24:00:00                        # Run time (hh:mm:ss)
#SBATCH -A IRI23004                        # Allocation name
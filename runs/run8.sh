#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 
#SBATCH --time=1:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=optimizers/sgd_CyclicLR3
#SBATCH --output=optimizers/sgd_CyclicLR3.out

module load python/intel/3.8.6
module load openmpi/intel/4.0.5

source ../venvs/dl/bin/activate
time python3 main.py  --exp optimizers/sgd_CyclicLR3 --opt sgd --lr_sched CyclicLR3 --data_augmentation --data_normalize --grad_clip 0.1
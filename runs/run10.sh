#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 
#SBATCH --time=1:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=adam_OneCycleLR
#SBATCH --output=adam_OneCycleLR.out

module load python/intel/3.8.6
module load openmpi/intel/4.0.5

source ../venvs/dl/bin/activate
time python3 main.py  --exp optimizers/adam_OneCycleLR --opt adam --lr_sched OneCycleLR --data_augmentation --data_normalize --grad_clip 0.1
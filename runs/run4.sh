#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 
#SBATCH --time=1:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=batch_size128_lr0.4
#SBATCH --output=batch_size128_lr0.4.out

module load python/intel/3.8.6
module load openmpi/intel/4.0.5

source ../venvs/dl/bin/activate
time python3 main.py  --exp batch_size_lr/batch_size128_lr0.4 --batch_size 128 --lr 0.4 --data_augmentation --data_normalize --grad_clip 0.1
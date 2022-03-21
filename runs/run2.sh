#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 
#SBATCH --time=1:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=vanilla_ResNet4_num_blocks[1, 1, 1, 1]_num_channels32
#SBATCH --output=vanilla_ResNet4_num_blocks[1, 1, 1, 1]_num_channels32.out

module load python/intel/3.8.6
module load openmpi/intel/4.0.5

source ../venvs/dl/bin/activate
time python3 main.py  --config resnet_configs/sunday_vanilla_ResNets4.yaml --resnet_architecture vanilla_ResNet4_num_blocks[1, 1, 1, 1]_num_channels32
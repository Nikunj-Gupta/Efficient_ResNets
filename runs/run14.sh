#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 
#SBATCH --time=1:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=vanilla_ResNet2_num_blocks2x1_num_channels128_conv5
#SBATCH --output=vanilla_ResNet2_num_blocks2x1_num_channels128_conv5.out

module load python/intel/3.8.6
module load openmpi/intel/4.0.5

source ../venvs/dl/bin/activate
time python3 main.py  --config resnet_configs/sunday_vanilla_ResNets2.yaml --resnet_architecture vanilla_ResNet2_num_blocks2x1_num_channels128_conv5
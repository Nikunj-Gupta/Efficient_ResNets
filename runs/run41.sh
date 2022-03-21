#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 
#SBATCH --time=1:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=good_ResNet3_num_blocks1x1x1_num_channels64_conv3
#SBATCH --output=good_ResNet3_num_blocks1x1x1_num_channels64_conv3.out

module load python/intel/3.8.6
module load openmpi/intel/4.0.5

source ../venvs/dl/bin/activate
time python3 main.py  --config resnet_configs/sunday_good_ResNets3.yaml --resnet_architecture good_ResNet3_num_blocks1x1x1_num_channels64_conv3
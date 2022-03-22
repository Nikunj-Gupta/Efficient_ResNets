#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 
#SBATCH --time=2:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=se_fulldrop_good_ResNet4_num_blocks2x1x1x1_squeeze_and_excitation0_drop0.8
#SBATCH --output=se_fulldrop_good_ResNet4_num_blocks2x1x1x1_squeeze_and_excitation0_drop0.8.out

module load python/intel/3.8.6
module load openmpi/intel/4.0.5

source ../venvs/dl/bin/activate
time python3 main.py  --config resnet_configs/se_fulldrop_good_ResNet4.yaml --resnet_architecture se_fulldrop_good_ResNet4_num_blocks2x1x1x1_squeeze_and_excitation0_drop0.8
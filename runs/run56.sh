#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 
#SBATCH --time=4:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=baseline_ResNet_gradclip0.1
#SBATCH --output=baseline_ResNet_gradclip0.1.out

module load python/intel/3.8.6
module load openmpi/intel/4.0.5

source ../venvs/dl/bin/activate
time python3 main.py  --config resnet_configs/sunday_ResNets.yaml --resnet_architecture baseline_ResNet_gradclip0.1
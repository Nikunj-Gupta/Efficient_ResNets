#!/bin/bash 
#SBATCH --nodes=3                        # requests 3 compute servers
#SBATCH --ntasks-per-node=2              # runs 2 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --time=1:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=resnet18
#SBATCH --output=resnet18.out

module load python/intel/3.8.6
module load openmpi/intel/4.0.5

source ../../venvs/dl/bin/activate
time python3 ../main.py 


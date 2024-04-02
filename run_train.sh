#!/bin/bash
#SBATCH --job-name=test-resnet
#SBATCH --account=project_2006419 
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1
module load tykky

export PATH="/scratch/project_2006419/aime/RGB_based_methods/ResNet101/container/bin:$PATH"

echo "start training"
srun python train.py 
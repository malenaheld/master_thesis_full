#!/bin/bash
#SBATCH --partition=intelsr_medium
#SBATCH --time=7:00:00
# SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

module load Python/3.11.5-GCCcore-13.2.0
module load Miniforge3
source ~/.bashrc
conda init bash
conda activate myenv



python prep_for_salmo.py

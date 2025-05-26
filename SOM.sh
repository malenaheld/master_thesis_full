#!/bin/bash
#SBATCH --partition=vlm_medium
#SBATCH --time=7:00:00
# SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

module load Python/3.11.5-GCCcore-13.2.0
module load Miniforge3
source ~/.bashrc
conda init bash
conda activate myenv

cd SOMcopy/
#python SOM_training.py
python SOM_clustering.py
cd ..

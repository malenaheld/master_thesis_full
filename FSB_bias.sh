#!/bin/bash
#SBATCH --array=1-100
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

module load CMake/3.26.3-GCCcore-12.3.0
module load pkg-config/0.29.2-GCCcore-11.2.0
module load GCC/13.2.0
module load GSL/2.7-GCC-12.3.0
module load CFITSIO/4.2.0-GCCcore-12.2.0
module load FFTW/3.3.10-GCC-12.3.0

# mkdir /lustre/scratch/data/s6maheld_hpc-data/trueVD/FSB_bias/
cd SALMO/
python t1.py $((SLURM_ARRAY_TASK_ID))
cd ..

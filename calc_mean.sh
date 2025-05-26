#!/bin/bash
#SBATCH --partition=vlm_devel
#SBATCH --time=0:05:00
# SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5


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
module load FFTW



python calc_mean.py /lustre/scratch/data/s6maheld_hpc-data/multiVD/ClsSALMO Cls
#python calc_mean.py /lustre/scratch/data/s6maheld_hpc-data/1024_2/ClsShells Cls
#python calc_mean.py /lustre/scratch/data/s6maheld_hpc-data/1024_2/ClsGLASS Cls
python calc_mean.py /lustre/scratch/data/s6maheld_hpc-data/multiVD/FsbSALMO FSB
python calc_mean.py /lustre/scratch/data/s6maheld_hpc-data/multiVD/sysrels systematic_relations

#python calc_mean.py /lustre/scratch/data/s6maheld_hpc-data/highVD/methods_deprojection FSB
#python calc_mean.py /lustre/scratch/data/s6maheld_hpc-data/highVD/methods_deprojection cross_terms

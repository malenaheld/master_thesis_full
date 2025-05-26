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


cd systematics_relations/
python get_sysrel.py 1 ../templates/m_ngal.fits sysrels_1systematic.data
python get_sysrel.py 3 ../templates/m_ngal.fits sysrels_3systematics.data
python get_sysrel.py 3 ../SOM/template_multiVD_nclust30.fits sysrels_3systematics_30clust.data
python get_sysrel.py 3 ../SOM/template_multiVD_nclust40.fits sysrels_3systematics_40clust.data
python get_sysrel.py 3 ../SOM/template_multiVD_nclust20.fits sysrels_3systematics_20clust.data
python get_sysrel.py 3 ../SOM/template_multiVD_nclust10.fits sysrels_3systematics_10clust.data
cd ..

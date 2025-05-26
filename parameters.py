import numpy as np


## general parameters
nside = 2048


## parameters for GLASS simulation
lmax = 3*nside + 1
ncorr = 3
dx = 140

## parameters for FSB
nbands = 8             ## how many bands?
lmax_filts = 2*nside    ## up to which l do you want the filters

## parameters for bandpowers of PCL
bins = 1    ## put False if you want linear bins
dell = 16   ## number of ells per bandpower if linear bins are used
b = np.array([   2,    4,    8,   16,   32,   64,  128,  256, 512, 1024, 2048, 4096, lmax-1]) 


## paths to store data   
path_ws = '/lustre/scratch/data/s6maheld_hpc-data/trueVD/'
path_glass = f'/lustre/scratch/data/s6maheld_hpc-data/{nside}/'
path_SOM = '' # f'/lustre/scratch/data/s6maheld_hpc-data/SOM/'

## maps
mask = 'DECALS_mask.fits'
depthmap = '../templates/m_nstar.fits'

## templates
temp_list = ['m_nstar.fits', 'm_completeness.fits', 'm_extinction.fits']
temp_min = [-1,-1,-1]

## parameters for the SOM
SOM_side = 54
sigma = 35
learning_rate = 0.1
n_epochs = 20
n_cluster_list = [30,40]


## other parameters
a = False # -0.0287
DoVD = True




#inputVDmap = 'template_multiVD.fits'
inputVDmap = depthmap
#inputVDmap = 'multi_mult_depthmap.fits'






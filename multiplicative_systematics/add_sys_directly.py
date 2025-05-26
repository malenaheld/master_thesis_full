import healpy as hp
import numpy as np
from astropy.io import fits
import pymaster as nmt
from tqdm import tqdm
import pandas as pd

def CreateGalaxyNumberDensityMap(file, mask, nside):
    ''' 
    Write a catalogue of galaxies with given right acension and declination to a healpy man that depicts the number density

    arguments:
        file: filename of the file containing the catalogue
        mask: healpy-map of the mask
        nside: desired nside of the number density map 
    '''
    
    catalogue = fits.open(file)[1].data

    RA = catalogue['ALPHA_J2000']
    DEC = catalogue['DELTA_J2000']

    
    pixind = hp.ang2pix(nside, RA, DEC, lonlat=True)
    npix = hp.nside2npix(nside)

    ## create map of galaxies
    galaxymap = np.bincount(pixind, minlength=npix)

    n_mean_gal = np.sum(galaxymap[np.where(mask!=0)[0]]) / len(np.where(mask!=0)[0])
    n_dens_gals = galaxymap/n_mean_gal

    
    return n_dens_gals

a = -1 
b = 0.94

nside = 1024

mask = hp.read_map('../DECALS_mask.fits')
temp = hp.read_map('../DECALS_DepthMap_fullsky.fits')

bedges = np.array([   2,    4,    8,   16,   32,   64,  128,  256,  512, 1024, 2048, 3*nside])
bins = nmt.NmtBin.from_edges(bedges[:-1], bedges[1:])
ells = bins.get_effective_ells()

results = {}

Nsim = 10
for i in tqdm(range(Nsim)):
    galmap0 = CreateGalaxyNumberDensityMap(f'/vol/aleph/data/mheld/salmo/SALMO_simulations/nonlinearbias/1024_m1/Catalogues/galCat_{i+1}_run2_type0.fits', mask, nside)

    f0 = nmt.NmtField(mask, [galmap0 - 1])
    cls0 = nmt.compute_full_master(f0, f0, bins)[0]

    ## add systematics:
    #galmap = galmap0 * (a*temp / b + 1)
    galmap = galmap0 + (a*temp/b)

    f_rew = nmt.NmtField(mask, [galmap / (a*temp / b + 1) -1])
    cls_rew = nmt.compute_full_master(f_rew, f_rew, bins)[0]

    ## calculate cls of the deprojected map and the contaminated map
    f1 = nmt.NmtField(mask, [galmap - 1])
    cls1 = nmt.compute_full_master(f1, f1, bins)[0]

    f_dep = nmt.NmtField(mask, [galmap  - 1], templates=[[temp]])
    cls_dep = nmt.compute_full_master(f_dep, f_dep, bins)[0]

    

    results[f'cls0: {i+1}'] = cls0
    results[f'cls1: {i+1}'] = cls1
    results[f'cls_dep: {i+1}'] = cls_dep
    results[f'cls_rew: {i+1}'] = cls_rew

results_mean = {}
results_mean['ells'] = ells
results_mean[f'cls0'] = np.mean([results[f'cls0: {i+1}'] for i in range(Nsim)], axis = 0)
results_mean[f'cls1'] = np.mean([results[f'cls1: {i+1}'] for i in range(Nsim)], axis = 0)
results_mean[f'cls_dep'] = np.mean([results[f'cls_dep: {i+1}'] for i in range(Nsim)], axis = 0)
results_mean[f'cls_rew'] = np.mean([results[f'cls_rew: {i+1}'] for i in range(Nsim)], axis = 0)

results_mean[f'cls0 unc'] = np.std([results[f'cls0: {i+1}'] for i in range(Nsim)], axis = 0) / np.sqrt(Nsim)
results_mean[f'cls1 unc'] = np.std([results[f'cls1: {i+1}'] for i in range(Nsim)], axis = 0) / np.sqrt(Nsim)
results_mean[f'cls_dep unc'] = np.std([results[f'cls_dep: {i+1}'] for i in range(Nsim)], axis = 0) / np.sqrt(Nsim)
results_mean[f'cls_rew unc'] = np.std([results[f'cls_rew: {i+1}'] for i in range(Nsim)], axis = 0) / np.sqrt(Nsim)

pd.DataFrame(results_mean).to_csv('add_sys_directly2.data')
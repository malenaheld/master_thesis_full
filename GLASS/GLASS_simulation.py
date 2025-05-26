import sys
from pathlib import Path

# Get the parent directory of the script
parent_dir = Path(__file__).resolve().parent.parent

# Add the parent directory to sys.path
sys.path.append(str(parent_dir))

import numpy as np
import healpy as hp
from astropy.io import fits
import glass.shells
import glass.fields
import argparse
import parameters as pars
import glass.ext.camb
import camb
from cosmology import Cosmology
import pymaster as nmt
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('taskID')

args = parser.parse_args()


ncorr = pars.ncorr
nside = pars.nside
dell = pars.dell
lmax = pars.lmax
dx = pars.dx
mask = pars.mask
path = pars.path_glass

# cosmology for the simulation
h = 0.7
Oc = 0.25
Ob = 0.05
ns = 0.965

# set up CAMB parameters for matter angular power spectrum
params = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2, NonLinear=camb.model.NonLinear_both)
params.InitPower.set_params(ns=ns)
#params.set_matter_power(kmax=10)

# get the cosmology from CAMB
cosmology = Cosmology.from_camb(params)

print('imported modules',flush=True)
### function to save matter shells in the correct format for salmo
def SaveForSALMO(filename, map):
    nside = hp.get_nside(map)
    npix = hp.nside2npix(nside)

    num_chunks = npix // 1024

    if npix % 1024 != 0:
        num_chunks += 1
        map = np.pad(map, (0, 1024 - (npix % 1024)), 'constant')

    format_map = map.reshape(num_chunks, 1024)

    col = fits.Column(name='signal', format='1024E', array=format_map)
    cols = fits.ColDefs([col])

    binary_table = fits.BinTableHDU.from_columns(cols)
    primary = fits.PrimaryHDU()

    hdulist = fits.HDUList([primary, binary_table])

    hdulist.writeto(filename, overwrite=True)
    return


# redshift bins from DELS
zb = np.load('../data_DELS/z.npy')
# redshift distribution
nz = np.load('../data_DELS/nz.npy')   ## weights according to the redshift distribution


# CAMB requires linear ramp for low redshifts
zb_cut = glass.shells.distance_grid(cosmology, 0,1, dx=dx)

ws = glass.shells.tophat_windows(zb_cut, weight=glass.ext.camb.camb_tophat_weight)
print('created weights',flush=True)


cls = np.load(f'../theory/cls_dx{dx}.npy')   ## load cls
print('loaded cls', flush=True)

cls = glass.fields.gaussian_gls(cls, ncorr=ncorr, lmax=lmax)

# compute Gaussian cls for lognormal fields with 3 correlated shells
gls = glass.fields.lognormal_gls(cls, ncorr=ncorr, lmax=lmax)
print('computed gls', flush=True)

# this generator will yield the matter fields in each shell
matter = glass.fields.generate_lognormal(gls, nside, ncorr=ncorr)
print('computed matter shells', flush=True)


npix = hp.nside2npix(nside)
matter_map = np.zeros_like(npix)
for i, delta_i in enumerate(matter):
   # restrict galaxy distribution to this shell
   z_i, dndz_i = glass.shells.restrict(zb, nz, ws[i])
   
   # integrate dndz to get the total galaxy density in this shell
   ngal = np.trapz(dndz_i, z_i)

   matter_map = matter_map + ngal*delta_i
   
   ## save matter shells to use for SALMO
   SaveForSALMO(f'{path}MatterShells/DECALS_denMap_{args.taskID}_run2_f1z{i+1}.fits', delta_i)
   print(f'saved matter shell {i+1}', flush=True)


hp.write_map(f'{path}MatterDensMaps/GLASS_map_{args.taskID}.fits', matter_map, overwrite=True)
print('Saved projected GLASS simulation', flush=True)

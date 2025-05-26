import sys
from pathlib import Path

# Get the parent directory of the script
parent_dir = Path(__file__).resolve().parent.parent

# Add the parent directory to sys.path
sys.path.append(str(parent_dir))

from astropy.io import fits
import numpy as np
import healpy as hp
import argparse
import parameters as pars

parser = argparse.ArgumentParser()

parser.add_argument('taskID')


args = parser.parse_args()

def CreateGalaxyNumberDensityMap(file, nside):
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
    
    ## redshift of galaxies
    z = np.array(catalogue['z_spec'])
    return galaxymap, z


nside = pars.nside
path = pars.path_ws

GalaxyDensity_type0, z0 = CreateGalaxyNumberDensityMap(f'{path}Catalogues/galCat_{args.taskID}_run2_type0.fits', nside)
hp.write_map(f'{path}GalMaps/GalCount_{args.taskID}_type0.fits', GalaxyDensity_type0, overwrite=True)


if pars.DoVD == True:
   GalaxyDensity_type1, z1 = CreateGalaxyNumberDensityMap(f'{path}Catalogues/galCat_{args.taskID}_run2_type1.fits', nside)
   hp.write_map(f'{path}GalMaps/GalCount_{args.taskID}_type1.fits', GalaxyDensity_type1, overwrite=True)


np.save(f'{path}redshifts/z0_{args.taskID}.npy', z0)
np.save(f'{path}redshifts/z1_{args.taskID}.npy', z1)


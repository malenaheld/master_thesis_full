## up/downgrade all maps to the nside wanted for the simulation
import healpy as hp
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

nside = 1024
nclust = 20

## templates
def resave(filein, fileout, nside, renorm=False):
    hp.write_map(fileout, hp.ud_grade(hp.read_map(filein), nside), overwrite=True)
    return 

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


## templates for deprojection
m_nstar = hp.ud_grade(hp.read_map('templates/m_nstar.fits'), nside)
m_completeness = hp.ud_grade(hp.read_map('templates/m_completeness.fits'), nside)
m_extinction = hp.ud_grade(hp.read_map('templates/m_extinction.fits'), nside)

## mask
mask = hp.ud_grade(hp.read_map('templates/Legacy_footprint_final_mask.fits'), nside) #'maps_for_salmo/DECALS_mask.fits', nside)

SaveForSALMO(f'maps_for_salmo/DECALS_mask.fits', mask)
mask[m_extinction > 0.3] = 0
SaveForSALMO(f'maps_for_salmo/DECALS_mask_multi.fits', mask)

## save masked and renormalized templates
SaveForSALMO(f'maps_for_salmo/m_nstar.fits', m_nstar)
SaveForSALMO(f'maps_for_salmo/m_completeness.fits', m_completeness)
SaveForSALMO(f'maps_for_salmo/m_extinction.fits', m_extinction)


## vd map (from SOM)
multiVD = hp.ud_grade(hp.read_map(f'SOM/template_multiVD_nclust{nclust}.fits'), nside)
SaveForSALMO(f'maps_for_salmo/template_multiVD.fits', (multiVD / np.mean(multiVD[mask==1]))*mask)

## vd map manually created
#SaveForSALMO('maps_for_salmo/DECALS_DepthMap_multi.fits', hp.ud_grade(hp.read_map('DECALS_DepthMap_multi.fits'), nside))

## vd map (only stars)
#SaveForSALMO('maps_for_salmo/DECALS_DepthMap_fullsky.fits', hp.ud_grade(hp.read_map('DECALS_DepthMap_fullsky.fits'), nside))

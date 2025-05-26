import pymaster as nmt
import numpy as np
import scipy.optimize as so
import healpy as hp
from astropy.io import fits
import matplotlib.pyplot as plt
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
    return galaxymap / np.mean(galaxymap[mask==1])

def linear(x, a, b):
    return a*x+b


def CalculateSystematics(ngal, nstar, bin_edges):
    nbins = len(bin_edges) - 1
    
    ngal_mean_arr = np.array([])
    
    for bin in range(nbins):
        min = bin_edges[bin]
        max = bin_edges[bin+1]
    
        inds = (nstar>min) & (nstar<max)

        ngal_mean_arr = np.append(ngal_mean_arr, np.nanmean(ngal[inds]))
    return ngal_mean_arr

def get_binedges(template, mask, nbins):
    ''' function that returns bin edges in order to have the same number of data points per bin, also returns the new indices of the temp sorted by values'''
    N_pix = len(template[np.where(mask!=0)])
    n_pix_per_bin = N_pix // nbins
    ## sort arrays by stellar density
    ind_sort = np.argsort(template[np.where(mask!=0)])
    temp_sort = template[np.where(mask!=0)][ind_sort]

    return ind_sort, temp_sort[0:(nbins+1)*n_pix_per_bin:n_pix_per_bin]


    
    
a_list = [-0.15, -0.287,-0.5,-1]
folder_list = ['1024_mp15/', '1024_high/', '1024_mp5/', '1024_m1/']

## choose linear bins
bmin = -.2
bmax = .6
nbins = 10
#bin_edges = np.linspace(bmin, bmax, nbins + 1)

temp = hp.read_map('../DECALS_DepthMap_fullsky.fits')
mask = hp.read_map('../DECALS_mask.fits')

nside = 1024
#Nsim = 50

ind_sort, bin_edges = get_binedges(temp, mask, nbins)

results_sims = {}
results_fit = {}
results_maps = {}

for a, folder in zip(a_list, folder_list):
    path = f'/vol/aleph/data/mheld/salmo/SALMO_simulations/nonlinearbias/{folder}'
    
    #files = [f for f in os.listdir(f'{path}Catalogues/') if 'type1' in f]
    #Nsim = len(files)
    #print(files)
    #inds = []
    #for file in files:
    #    inds.append(np.uint(re.findall(r'\d+', file)[0]))
        
    #inds = inds[:20]
    #files = files[:20]
    #for ind, file in zip(inds, files):
        #print(file)
        
        
    galmap = CreateGalaxyNumberDensityMap(f'{path}Catalogues/galCat_1_run2_type1.fits', mask, nside)
    results_maps[f'Sim with VD: {a}'] = galmap*mask
    
    
    ## systematics
    n_gals_VD = CalculateSystematics(galmap[np.where(mask!=0)][ind_sort], temp[np.where(mask!=0)][ind_sort], bin_edges)
    #results[f'with VD {ind}'] = n_gals_VD
    results_sims[f'with VD: {a}'] = n_gals_VD
    
    
    ## deproject the map using mode deprojection
    f_dep = nmt.NmtField(mask, [galmap], templates=[[temp]])
    m_dep = f_dep.get_maps()[0]
    results_maps[f'Deprojected map: {a}'] = m_dep
    
    ## systematics
    n_gals_modeD = CalculateSystematics(m_dep[np.where(mask!=0)], temp[np.where(mask!=0)], bin_edges)
    #results[f'mode deprojection {ind}'] = n_gals_modeD
    results_sims[f'mode deprojection: {a}'] = n_gals_modeD
    
    
    
    ## deproject the map using DES-Y1 method
    f_rew = nmt.NmtField(mask, [galmap / (a*temp/0.94 + 1)])
    m_rew = f_rew.get_maps()[0]
    results_maps[f'Reweighted map: {a}'] = m_rew
    
    ## systematics
    n_gals_reweight = CalculateSystematics(m_rew[np.where(mask!=0)], temp[np.where(mask!=0)], bin_edges)
    #results[f'DES Y1 method {ind}'] = n_gals_reweight
    results_sims[f'DES Y1 method: {a}'] = n_gals_reweight
        
    # ngals1_arr  = np.array([ngals[f'with VD {i}'] for i in inds])
    # ngals_dep_arr  = np.array([ngals[f'mode deprojection {i}'] for i in inds])
    # ngals_rew_arr  = np.array([ngals[f'DES Y1 method {i}'] for i in inds])
    

    results_sims[f'stars'] = (bin_edges[1:] + bin_edges[:-1]) / 2
    #n_gals_VD = np.mean([results[f'with VD {i}'] for i in inds], axis=0)
    #n_gals_modeD = np.mean([results[f'mode deprojection {i}'] for i in inds], axis=0)
    #n_gals_reweight = np.mean([ngals[f'DES Y1 method {i}'] for i in inds], axis=0)
    
    #fig = plt.figure()    
    pars, cov = so.curve_fit(linear, xdata=results_sims[f'stars'], ydata=n_gals_VD) #, sigma=unc)
    errors = np.sqrt(np.diag(cov))
    

    print(f'for a = {a}:\n systematics relation with VD:')
    print(f'f(x) = ({round(pars[0]*0.94, 5)} +- {round(errors[0]*0.94, 5)})* x + ({round(pars[1]*0.94, 3)} +- {round(errors[1]*0.94, 3)})')
    
    results_fit[f'with VD: {a}, a'] = pars[0]
    results_fit[f'with VD: {a}, a_unc'] = errors[0]
    results_fit[f'with VD: {a}, b'] = pars[1]
    results_fit[f'with VD: {a}, b_unc'] = errors[1]

    pars, cov = so.curve_fit(linear, xdata=results_sims[f'stars'], ydata=n_gals_modeD) #, sigma=unc)
    errors = np.sqrt(np.diag(cov))

    print(f'systematics relation after mode deprojection:')
    print(f'f(x) = ({round(pars[0]*0.94, 5)} +- {round(errors[0]*0.94, 5)})* x + ({round(pars[1]*0.94, 3)} +- {round(errors[1]*0.94, 3)})')
    
    results_fit[f'mode deprojection: {a}, a'] = pars[0]
    results_fit[f'mode deprojection: {a}, a_unc'] = errors[0]
    results_fit[f'mode deprojection: {a}, b'] = pars[1]
    results_fit[f'mode deprojection: {a}, b_unc'] = errors[1]
    
    pars, cov = so.curve_fit(linear, xdata=results_sims[f'stars'], ydata=n_gals_reweight) #, sigma=unc)
    errors = np.sqrt(np.diag(cov))

    print(f'systematics relation after DES-Y1 method:')
    print(f'f(x) = ({round(pars[0]*0.94, 5)} +- {round(errors[0]*0.94, 5)})* x + ({round(pars[1]*0.94, 3)} +- {round(errors[1]*0.94, 3)})')

    results_fit[f'DES Y1 method: {a}, a'] = pars[0]
    results_fit[f'DES Y1 method: {a}, a_unc'] = errors[0]
    results_fit[f'DES Y1 method: {a}, b'] = pars[1]
    results_fit[f'DES Y1 method: {a}, b_unc'] = errors[1]


pd.DataFrame(results_sims).to_csv('nstarVSngal_sims.data')
pd.DataFrame([results_fit]).to_csv('nstarVSngal_fitparams.data')
pd.DataFrame(results_maps).to_csv('maps.data')
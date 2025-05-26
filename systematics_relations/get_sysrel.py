import numpy as np
import healpy as hp
import pandas as pd
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('Nsys')
parser.add_argument('inputmap')
parser.add_argument('file')

args = parser.parse_args()
Nsys = int(args.Nsys)  ## number of systematics templates (either 1 or 3)
inputmap = args.inputmap
file = args.file

print(inputmap)
def CalculateSystematics(ngal, nstar, bin_edges):
    ''' function that calculates the temp vs ngal relation given specified bin edges'''
    nbins = len(bin_edges) - 1
    
    ngal_mean_arr = np.array([])
    ngal_unc_arr = np.array([])
    temp_vals_arr = np.array([])
    for bin in range(nbins):
        min = bin_edges[bin]
        max = bin_edges[bin+1]
    
        inds = (nstar>min) & (nstar<max)

        ngal_mean_arr = np.append(ngal_mean_arr, np.nanmean(ngal[inds]))
        ngal_unc_arr = np.append(ngal_unc_arr, np.nanstd(ngal[inds])/ np.sqrt(len(ngal[inds])))
        temp_vals_arr = np.append(temp_vals_arr, np.nanmean(nstar[inds]))
    return ngal_mean_arr, ngal_unc_arr, temp_vals_arr

def get_binedges(template, mask, nbins):
    ''' function that returns bin edges in order to have the same number of data points per bin, also returns the new indices of the temp sorted by values'''
    N_pix = len(template[np.where(mask!=0)])
    n_pix_per_bin = N_pix // nbins
    ## sort arrays by stellar density
    ind_sort = np.argsort(template[np.where(mask!=0)])
    temp_sort = template[np.where(mask!=0)][ind_sort]

    return ind_sort, temp_sort[0:(nbins+1)*n_pix_per_bin:n_pix_per_bin]


nside = 1024

## read in templates
m_nstar = hp.ud_grade(hp.read_map('../maps_for_salmo/m_nstar.fits'), nside)
m_completeness = hp.ud_grade(hp.read_map('../maps_for_salmo/m_completeness.fits'), nside)
m_completeness[m_completeness < 0.86] = np.nan
m_extinction = hp.ud_grade(hp.read_map('../maps_for_salmo/m_extinction.fits'), nside)

## define templates
if Nsys==1:
    temps = np.array([m_nstar])
    temps_names = ['nstar']
elif Nsys==3:
    temps = np.array([m_nstar, m_completeness, m_extinction])
    temps_names = ['nstar', 'completeness', 'extinction']

mask = hp.ud_grade(hp.read_map('../maps_for_salmo/DECALS_mask.fits'), nside)


map = hp.ud_grade(hp.read_map(inputmap), nside)
map = map / np.nanmean(map[mask==1])


sysrels = {}
for i, (name, temp) in enumerate(zip(temps_names, temps)):
    bmin = np.nanmin(temp[mask!=0])
    bmax = np.nanmax(temp[mask!=0])
    nbins = 10
    
    bin_edges = np.linspace(bmin, bmax, nbins + 1)
    ind_sort = np.arange(len(temp[mask!=0]))
    #ind_sort, bin_edges = get_binedges(temp, mask, nbins)
    print(bin_edges)
    
    n_gals, n_gals_unc, temp_binned = CalculateSystematics(map[np.where(mask!=0)][ind_sort], temp[np.where(mask!=0)][ind_sort], bin_edges)
    
    sysrels[f'{name}: temp_vals'] = temp_binned
    sysrels[f'{name}: ngal'] = n_gals
    sysrels[f'{name}: ngal unc'] = n_gals_unc
    
pd.DataFrame(sysrels).to_csv(file)

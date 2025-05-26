import numpy as np
import scipy.optimize as so

def CalculateSystematics(ngal, nstar, bin_edges):
    ''' function that calculates the temp vs ngal relation given specified bin edges'''
    nbins = len(bin_edges) - 1

    ngal_mean_arr = np.array([])
    temp_vals_arr = np.array([])
    for bin in range(nbins):
        min = bin_edges[bin]
        max = bin_edges[bin+1]
        if min == max:
            pass
        else:
            inds = (nstar>=min) & (nstar<max)
            ngal_mean_arr = np.append(ngal_mean_arr, np.nanmean(ngal[inds]))
            temp_vals_arr = np.append(temp_vals_arr, np.nanmean(nstar[inds]))
    return ngal_mean_arr, temp_vals_arr

def get_binedges(template, mask, nbins):
    ''' function that returns bin edges in order to have the same number of data points per bin, also returns the new indices of the temp sorted by values'''

    N_pix = len(template[np.where(mask!=0)])
    n_pix_per_bin = N_pix // nbins
    ## sort arrays by stellar density
    ind_sort = np.argsort(template[np.where(mask!=0)])
    temp_sort = template[np.where(mask!=0)][ind_sort]

    return ind_sort, temp_sort[0:(nbins+1)*n_pix_per_bin:n_pix_per_bin]

def fit(x, a, b): 
    return a + b*x 

def reweight_map(map_in, mask, templates, a_sys=False):
    ''' this function should take a contaminated map and then reweight it by iteratively fitting linear functions to the systematic relations for all templates'''
    map = map_in.copy()
    for rep in [1]:
        for i, temp in enumerate(templates[:-1]):
            temp_masked = temp[0][mask!=0]
            if a_sys:
                map = map / (a_sys / 0.94 *temp +1)
            else:
                nbins = 10
                ind_sort = np.arange(len(temp_masked))
                bin_edges = np.linspace(np.min(temp_masked), np.max(temp_masked), nbins+1)

                ## calculate the systematic relation
                ngals, temp_vals = CalculateSystematics(map[np.where(mask!=0)][ind_sort], temp_masked[ind_sort], bin_edges)

                ## fit a linear function
                pars, cov = so.curve_fit(fit, xdata=temp_vals[np.where(np.isnan(temp_vals)==False)], ydata=ngals[np.where(np.isnan(ngals)==False)])
                
                ## reweight map
                map = map / (fit(temp[0], *pars) / pars[0])
    return map

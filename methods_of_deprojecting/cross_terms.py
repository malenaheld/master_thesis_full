import sys
from pathlib import Path

# Get the parent directory of the script
parent_dir = Path(__file__).resolve().parent.parent

# Add the parent directory to sys.path
sys.path.append(str(parent_dir))

## import necessary modules
import FSB as fsb
import healpy as hp
import pymaster as nmt
import numpy as np
import pandas as pd
import parameters as pars
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('taskID')
args = parser.parse_args()


## read in parameters from the parameter file
nside = pars.nside
nbands = pars.nbands
lmax = pars.lmax_filts
deltaell = pars.dell
dell = lmax // nbands

path_ws = pars.path_ws
## get the filter functions for calculating the FSB
filters = fsb.get_filters(nbands, lmax)

## define the bins
bins = nmt.NmtBin.from_nside_linear(nside, deltaell)
ells = bins.get_effective_ells()
nbins = len(ells)

def read_map(file, nside):
    '''
    function that reads a map and directly converts it to the nside that is used
    '''
    return  hp.ud_grade(hp.read_map(file), nside)

def CreateNumDens(galmap, mask):
    n_mean = np.sum(galmap[np.where(mask!=0)[0]]) / len(np.where(mask!=0)[0])
    n_dens_gals = galmap/n_mean
    return n_dens_gals

def get_fields_from_map_wtemp_filt(delta, mask, fls, nside, templates=None, sub_mean=True, iter=3):
    """ Returns NaMaster fields from a given map.

    Args:
        delta: input map.
        mask: mask associated with this field. Note that we assume the
            mask to be binary.
        sub_mean: ensure that all filtered-squared maps have zero mean?
        iter: number of Jacobi iterations when computing SHTs.

    Returns: dictionary of NaMaster fields. 'fN' denotes the original
        overdensity field, while 'f1', 'f2', etc. correspond to the
        filtered-squared maps of the different filters.
    """
    flds = {}
    # Mask things for safety.
    # Note we're assuming mask to be binary at this stage, otherwise this is wrong.

    mp = delta*mask
    if sub_mean:
        mp = mp - mask * np.sum(mp*mask)/np.sum(mask)
    flds['fN'] = nmt.NmtField(mask, [mp], templates=templates, n_iter=iter)
    # Alm of the original map
    alm = hp.map2alm(mp, iter=iter)

    template = fsb.get_filtered_field(templates[0][0], mask, fls, nside, sub_mean=True, iter=3)

    # Filtered-squared maps for each filter
    mp_filt_sq = np.array([hp.alm2map(hp.almxfl(alm, fl), nside)**2 for fl in fls])
    if sub_mean:
        mp_filt_sq = np.array([m-mask*np.sum(m*mask)/np.sum(mask)
                               for m in mp_filt_sq])
    for i, m in enumerate(mp_filt_sq):
        flds[f'f{i}'] = nmt.NmtField(mask, [m], n_iter=iter, templates=[[template[f'f{i}'].get_maps()[0]**2]])

    return flds

def get_fields_from_map_wtemp_unfilt(delta, mask, fls, nside, templates=None, sub_mean=True, iter=3):
    """ Returns NaMaster fields from a given map.

    Args:
        delta: input map.
        mask: mask associated with this field. Note that we assume the
            mask to be binary.
        sub_mean: ensure that all filtered-squared maps have zero mean?
        iter: number of Jacobi iterations when computing SHTs.

    Returns: dictionary of NaMaster fields. 'fN' denotes the original
        overdensity field, while 'f1', 'f2', etc. correspond to the
        filtered-squared maps of the different filters.
    """
    flds = {}
    # Mask things for safety.
    # Note we're assuming mask to be binary at this stage, otherwise this is wrong.

    mp = delta*mask
    if sub_mean:
        mp = mp - mask * np.sum(mp*mask)/np.sum(mask)
    flds['fN'] = nmt.NmtField(mask, [mp], templates=templates, n_iter=iter)
    # Alm of the original map
    alm = hp.map2alm(mp, iter=iter)

    #template = fsb.get_filtered_field(templates[0][0], mask, fls, nside, sub_mean=True, iter=3)

    # Filtered-squared maps for each filter
    mp_filt_sq = np.array([hp.alm2map(hp.almxfl(alm, fl), nside)**2 for fl in fls])
    if sub_mean:
        mp_filt_sq = np.array([m-mask*np.sum(m*mask)/np.sum(mask)
                               for m in mp_filt_sq])
    for i, m in enumerate(mp_filt_sq):
        flds[f'f{i}'] = nmt.NmtField(mask, [m], n_iter=iter, templates=[[templates[0][0]**2]])

    return flds

path = f'{pars.path_ws}GalMaps'

## mask
MaskSalmo = read_map(f'../maps_for_salmo/{pars.mask}', nside)

## variable depth map
depth_map = read_map(f'{pars.inputVDmap}', nside)
depth_map = depth_map -np.mean(depth_map[MaskSalmo==1])


## define the workspace
wsp = nmt.NmtWorkspace()
f = nmt.NmtField(MaskSalmo, None, spin=0)
wsp.compute_coupling_matrix(f,f, bins)


## map of the galaxy count (output from SALMO)
galmap1 = CreateNumDens(read_map(f'{path}/GalCount_{args.taskID}_type1.fits', nside), MaskSalmo)
galmap0 = CreateNumDens(read_map(f'{path}/GalCount_{args.taskID}_type0.fits', nside), MaskSalmo)
#galmap1 = galmap0 - 0.287*depth_map


a_obs = fsb.get_fields_from_map(galmap1-1, MaskSalmo, filters, nside)
a_true = fsb.get_fields_from_map(galmap0-1, MaskSalmo, filters, nside)
a_clean_unfilt = get_fields_from_map_wtemp_filt(galmap1-1, MaskSalmo, filters, nside, templates=[[depth_map]])
#a_clean_unfilt = fsb.get_fields_from_map_wtemp2(galmap1, MaskSalmo, filters, nside, templates=[[depth_map]])

#a_diff_true = nmt.NmtField(MaskSalmo, [a_obs[f'fN'].get_maps()[0] - a_true[f'fN'].get_maps()[0]])
#a_diff_clean = nmt.NmtField(MaskSalmo, [a_obs[f'fN'].get_maps()[0] - a_clean_filt[f'fN'].get_maps()[0]])

#f = fsb.get_fields_from_map(depth_map, MaskSalmo, filters, nside)

results = {}
for j in range(nbands):
#    a_sq_diff_true = nmt.NmtField(MaskSalmo, [a_obs[f'f{j}'].get_maps()[0] - a_true[f'f{j}'].get_maps()[0]])
#    a_sq_diff_clean_filt = nmt.NmtField(MaskSalmo, [a_obs[f'f{j}'].get_maps()[0] - a_clean_filt[f'f{j}'].get_maps()[0]])
#    a_sq_diff_clean_unfilt = nmt.NmtField(MaskSalmo, [a_obs[f'f{j}'].get_maps()[0] - a_clean_unfilt[f'f{j}'].get_maps()[0]])
#    a_sq_diff_bias_filt = nmt.NmtField(MaskSalmo, [a_clean_filt[f'f{j}'].get_maps()[0] - a_true[f'f{j}'].get_maps()[0]])
    a_sq_diff_bias_unfilt = nmt.NmtField(MaskSalmo, [a_clean_unfilt[f'f{j}'].get_maps()[0] - a_true[f'f{j}'].get_maps()[0]])
    
    #diff2TrueXclean = fsb.get_cl(a_sq_diff_true, a_clean_filt[f'fN'], wsp)[0]
    #diff2TrueXtrue = fsb.get_cl(a_sq_diff_true, a_true[f'fN'], wsp)[0]
    #diff2TrueXdiffTrue = fsb.get_cl(a_sq_diff_true, a_diff_true, wsp)[0]
    #clean2filtXtrue = fsb.get_cl(a_clean_filt[f'f{j}'], a_true[f'fN'], wsp)[0]
    #clean2unfiltXtrue = fsb.get_cl(a_clean_unfilt[f'f{j}'], a_true[f'fN'], wsp)[0]
    #clean2filtXdiffTrue = fsb.get_cl(a_clean_filt[f'f{j}'], a_diff_true, wsp)[0]
    #clean2unfiltXdiffTrue = fsb.get_cl(a_clean_unfilt[f'f{j}'], a_diff_true, wsp)[0]

    #diff2cleanFiltXclean = fsb.get_cl(a_sq_diff_clean_filt, a_clean_filt[f'fN'], wsp)[0]
    #diff2cleanFiltXtrue = fsb.get_cl(a_sq_diff_clean_filt, a_true[f'fN'], wsp)[0]
    #diff2cleanFiltXdiffClean = fsb.get_cl(a_sq_diff_clean_filt, a_diff_clean, wsp)[0]
    #clean2filtXdiffClean = fsb.get_cl(a_clean_filt[f'f{j}'], a_diff_clean, wsp)[0]

    #diff2cleanUnfiltXclean = fsb.get_cl(a_sq_diff_clean_unfilt, a_clean_filt[f'fN'], wsp)[0]
    #diff2cleanUnfiltXtrue = fsb.get_cl(a_sq_diff_clean_unfilt, a_true[f'fN'], wsp)[0]
    #diff2cleanUnfiltXdiffClean = fsb.get_cl(a_sq_diff_clean_unfilt, a_diff_clean, wsp)[0]
    #clean2unfiltXdiffClean = fsb.get_cl(a_clean_unfilt[f'f{j}'], a_diff_clean, wsp)[0]
    
    #clean2unfiltXclean = fsb.get_cl(a_clean_unfilt[f'f{j}'], a_clean_filt[f'fN'], wsp)[0]
    #clean2filtXclean = fsb.get_cl(a_clean_filt[f'f{j}'], a_clean_filt[f'fN'], wsp)[0]
    
    #clean2unfiltXobs = fsb.get_cl(a_clean_unfilt[f'f{j}'], a_obs[f'fN'], wsp)[0]
    #clean2filtXobs = fsb.get_cl(a_clean_filt[f'f{j}'], a_obs[f'fN'], wsp)[0]
    
    #clean2filtXf = fsb.get_cl(a_clean_filt[f'f{j}'], f[f'fN'], wsp)[0]
    ##depbias2FiltXdiffClean = fsb.get_cl(a_sq_diff_bias_filt, a_diff_clean, wsp)[0]
    #depbias2UnfiltXdiffClean = fsb.get_cl(a_sq_diff_bias_unfilt, a_diff_clean, wsp)[0]
    #obsXdiffClean = fsb.get_cl(a_obs[f'f{j}'], a_diff_clean, wsp)[0]
    
    #trueXdiffclean = fsb.get_cl(a_true[f'f{j}'], a_diff_clean, wsp)[0]
    #depbias2FiltXclean = fsb.get_cl(a_sq_diff_bias_filt, a_clean_filt[f'fN'], wsp)[0]
    depbias2UnfiltXclean = fsb.get_cl(a_sq_diff_bias_unfilt, a_clean_unfilt[f'fN'], wsp)[0]
    
    
    #obs2Xobs = fsb.get_cl(a_obs[f'f{j}'], a_obs[f'fN'], wsp)[0]
    #obs2Xclean = fsb.get_cl(a_obs[f'f{j}'], a_clean_filt[f'fN'], wsp)[0]
    #obs2Xtrue = fsb.get_cl(a_obs[f'f{j}'], a_true[f'fN'], wsp)[0]
    
    true2Xobs = fsb.get_cl(a_true[f'f{j}'], a_obs[f'fN'], wsp)[0]
    true2Xclean = fsb.get_cl(a_true[f'f{j}'], a_clean_unfilt[f'fN'], wsp)[0]
    true2Xtrue = fsb.get_cl(a_true[f'f{j}'], a_true[f'fN'], wsp)[0]

    #f2xclean = fsb.get_cl(f[f'f{j}'], a_clean_filt[f'fN'], wsp)[0]

    #results[f'(a^2_obs - a^2_true)xa_c bin{j}'] = diff2TrueXclean
    #results[f'(a^2_obs - a^2_true)xa_true bin{j}'] = diff2TrueXtrue
    #results[f'(a^2_obs - a^2_true)x(a_obs-a_true) bin{j}'] = diff2TrueXdiffTrue
    #results[f'a^2_cxa_true filtered bin{j}'] = clean2filtXtrue
    #results[f'a^2_cxa_true unfiltered bin{j}'] = clean2unfiltXtrue
    #results[f'a_c^2x(a_obs - a_true) filtered bin{j}'] = clean2filtXdiffTrue
    #results[f'a_c^2x(a_obs - a_true) unfiltered bin{j}'] = clean2unfiltXdiffTrue

    #results[f'(a^2_obs - a^2_c)xa_c filtered bin{j}'] = diff2cleanFiltXclean
    #results[f'(a^2_obs - a^2_c)xa_true filtered bin{j}'] = diff2cleanFiltXtrue
    #results[f'(a^2_obs - a^2_c)x(a_obs-a_c) filtered bin{j}'] = diff2cleanFiltXdiffClean
    #results[f'a_c^2x(a_obs - a_c) filtered bin{j}'] = clean2filtXdiffClean

    #results[f'(a^2_obs - a^2_c)xa_c unfiltered bin{j}'] = diff2cleanUnfiltXclean
    #results[f'(a^2_obs - a^2_c)xa_true unfiltered bin{j}'] = diff2cleanUnfiltXtrue
    #results[f'(a^2_obs - a^2_c)x(a_obs-a_c) unfiltered bin{j}'] = diff2cleanUnfiltXdiffClean
    #results[f'a_c^2x(a_obs - a_c) unfiltered bin{j}'] = clean2unfiltXdiffClean
    
    #results[f'a_c^2xa_c unfiltered bin{j}'] = clean2unfiltXclean
    #results[f'a_c^2xa_c filtered bin{j}'] = clean2filtXclean
    
    #results[f'a_c^2xa_obs unfiltered bin{j}'] = clean2unfiltXobs
    #results[f'a_c^2xa_obs filtered bin{j}'] = clean2filtXobs
    
    #results[f'a_c^2xf filtered bin{j}'] = clean2filtXf
    #results[f'(a_c^2 - a_true^2)x(a_obs - a_clean) filtered bin{j}'] = depbias2FiltXdiffClean
    #results[f'(a_c^2 - a_true^2)x(a_obs - a_clean) unfiltered bin{j}'] = depbias2UnfiltXdiffClean
    #results[f'(a_obs^2x(a_obs - a_clean) filtered bin{j}'] = obsXdiffClean
    
    #results[f'(a_c^2 - a_true^2)x(a_obs - a_clean) filtered bin{j}'] = depbias2FiltXclean
    #results[f'(a_c^2 - a_true^2)x(a_obs - a_clean) unfiltered bin{j}'] = depbias2UnfiltXclean
    #results[f'(a_c^2 - a_true^2)xa_clean filtered bin{j}'] = depbias2FiltXclean
    results[f'(a_c^2 - a_true^2)xa_clean unfiltered bin{j}'] = depbias2UnfiltXclean
    
    
    #results[f'a_obs^2xa_obs unfiltered bin{j}'] = obs2Xobs
    #results[f'a_obs^2xa_clean unfiltered bin{j}'] = obs2Xclean
    #results[f'a_obs^2xa_true unfiltered bin{j}'] = obs2Xtrue
    
    results[f'a_true^2xa_obs unfiltered bin{j}'] = true2Xobs
    results[f'a_true^2xa_clean unfiltered bin{j}'] = true2Xclean
    results[f'a_true^2xa_true unfiltered bin{j}'] = true2Xtrue

    #results[f'f^2xa_clean bin{j}'] = f2xclean
df = pd.DataFrame(results)
df.to_csv(f'{path_ws}methods_deprojection/cross_terms_{args.taskID}.data')

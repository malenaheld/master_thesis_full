import sys
from pathlib import Path

# Get the parent directory of the script
parent_dir = Path(__file__).resolve().parent.parent

# Add the parent directory to sys.path
sys.path.append(str(parent_dir))

import pymaster as nmt   ### Namaster
import healpy as hp
import numpy as np
import FSB as fsb
import pandas as pd
import argparse
import parameters as pars
import scipy.optimize as so

## functions for calculating systematics and reweighting
import systematics as sys


parser = argparse.ArgumentParser()

parser.add_argument('taskID')

args = parser.parse_args()

# other parameters
deltal = pars.dell
nside = pars.nside
lmax = pars.lmax_filts
mask = pars.mask
path = pars.path_ws
a_sys = pars.a
nbands = pars.nbands
dell = lmax // nbands
filters = fsb.get_filters(nbands, lmax)

### Mask used in SALMO simulation
MaskSALMO = hp.read_map(f'../maps_for_salmo/{mask}')
MaskSALMO = hp.ud_grade(MaskSALMO, nside)

MaskFullsky = np.ones_like(MaskSALMO)


## define bins
if pars.bins == None:
   bins = nmt.NmtBin.from_nside_linear(nside, deltal)
else:
   bins = nmt.NmtBin.from_edges(pars.b[:-1], pars.b[1:])
ell_arr = bins.get_effective_ells()   ## l's corresponding to the bins
nbins = bins.get_n_bands()

## workspace
fmask = nmt.NmtField(MaskSALMO, None, spin=0)
wsp_mask = nmt.NmtWorkspace()
wsp_mask.compute_coupling_matrix(fmask, fmask, bins)

ffull = nmt.NmtField(MaskFullsky, None, spin=0)
wsp_full = nmt.NmtWorkspace()
wsp_full.compute_coupling_matrix(ffull, ffull, bins)

def CreateNumDens(galmap, mask):
    nmean = np.sum(galmap[np.where(mask==1)[0]]) / len(np.where(mask==1)[0])
    n_dens_gals = galmap/nmean
    print(nmean, flush=True)
    return n_dens_gals, nmean

def RemoveShotNoiseCL(cls, galmap, mask, nside, wsp):
    nmean = np.sum(galmap) / np.sum(mask)
    mMask = np.mean(mask)
    OmPix = hp.nside2pixarea(nside)

    cls_noise = OmPix * mMask / nmean * np.ones(3*nside)
    cls_noise = wsp.decouple_cell([cls_noise])[0]

    return cls - cls_noise



temp_list = pars.temp_list
temp_min = pars.temp_min
temp_max = pars.temp_max

m_nstar = hp.ud_grade(hp.read_map(f'../templates/{temp_list[0]}'), nside)
m_completeness = hp.ud_grade(hp.read_map(f'../templates/{temp_list[1]}'), nside)
m_extinction = hp.ud_grade(hp.read_map(f'../templates/{temp_list[2]}'), nside)


templates = [[m_nstar], [m_completeness], [m_extinction]]
#templates = [[m_nstar]]

for i, temp in enumerate(templates):
    temp[0][np.isnan(temp[0])] = 0
    temp[0][temp[0] < temp_min[i]] = temp_min[i]
    temp[0][temp[0] > temp_max[i]] = temp_max[i]
    temp[0] = temp[0] - np.mean(temp[0][MaskSALMO==1])

GalaxyDensity_type0, nmean = CreateNumDens(hp.ud_grade(hp.read_map(f'{path}GalMaps/GalCount_{args.taskID}_type0.fits'), nside), MaskSALMO)
f0 = nmt.NmtField(MaskSALMO, [GalaxyDensity_type0-1])

cls0 = nmt.compute_full_master(f0, f0, bins)[0]
cls0_rmSN = RemoveShotNoiseCL(cls0, GalaxyDensity_type0*MaskSALMO*nmean, MaskSALMO, nside, wsp_mask)

results_Cls = {}
results_Cls['ell'] = ell_arr
results_Cls['Cl_NoVD'] = cls0
results_Cls['Cl_NoVD_rmshotnoise'] = cls0_rmSN


if pars.DoVD == True:
    GalaxyDensity_type1, mean1 = CreateNumDens(hp.ud_grade(hp.read_map(f'{path}GalMaps/GalCount_{args.taskID}_type1.fits'), nside), MaskSALMO)
    f1 = nmt.NmtField(MaskSALMO, [GalaxyDensity_type1-1])
    
    cls1 = nmt.compute_full_master(f1, f1, bins)[0]
    cls1_rmSN = RemoveShotNoiseCL(cls1, GalaxyDensity_type1*MaskSALMO*mean1, MaskSALMO, nside, wsp_mask)


    ## deprojecting
    f_dep = nmt.NmtField(MaskSALMO, [GalaxyDensity_type1-1], templates=templates)
    m_dep = (f_dep.get_maps()[0] + 1) * mean1
    cls_dep = nmt.compute_full_master(f_dep, f_dep, bins)[0]
    cls_dep_rmSN = RemoveShotNoiseCL(cls_dep, m_dep*MaskSALMO, MaskSALMO, nside, wsp_mask)
    
    ## reweighting
    m_rew = sys.reweight_map(GalaxyDensity_type1*mean1, MaskSALMO, templates, a_sys=a_sys)
    f_rew = nmt.NmtField(MaskSALMO, [m_rew/np.mean(m_rew[MaskSALMO==1]) - 1], templates=[[np.ones_like(m_nstar)]])
    cls_rew = nmt.compute_full_master(f_rew, f_rew, bins)[0]
    cls_rew_rmSN = RemoveShotNoiseCL(cls_rew, m_rew*MaskSALMO, MaskSALMO, nside, wsp_mask)
    
    ## save results
    results_Cls['Cl_VD'] = cls1
    results_Cls['Cl_VD_rmshotnoise'] = cls1_rmSN
    results_Cls['Cl_Deprojected'] = cls_dep
    results_Cls['Cl_Deprojected_rmshotnoise'] = cls_dep_rmSN
    results_Cls['Cl_Reweighted'] = cls_rew
    results_Cls['Cl_Reweighted_rmshotnoise'] = cls_rew_rmSN

## save Cls
df_Cls = pd.DataFrame(results_Cls)
df_Cls.to_csv(f'{path}ClsSALMO/Cls_{args.taskID}.data')
print('Saved Cls', flush=True)



ff0 = fsb.get_fields_from_map(GalaxyDensity_type0-1, MaskSALMO, filters, nside)
ff0_full = fsb.get_fields_from_map((GalaxyDensity_type0-1)*MaskSALMO, MaskFullsky, filters, nside)

results_FSB = {}
results_FSB['ell'] = ell_arr
if pars.DoVD == True:
    ff1 = fsb.get_fields_from_map(GalaxyDensity_type1-1, MaskSALMO, filters, nside)
    ff_dep = fsb.get_fields_from_map_wtemp(GalaxyDensity_type1-1, MaskSALMO, filters, nside, templates=templates)
    ff_dep_full = fsb.get_fields_from_map_wtemp((GalaxyDensity_type1-1)*MaskSALMO, MaskFullsky, filters, nside, templates=templates)
    ff_rew = fsb.get_fields_from_map_wtemp(m_rew/np.mean(m_rew[MaskSALMO==1]), MaskSALMO, filters, nside)

for j in range(nbands):
    FSB0 = fsb.get_cl(ff0['fN'], ff0[f'f{j}'], wsp_mask)[0]
    FSB0_full = fsb.get_cl(ff0_full['fN'], ff0_full[f'f{j}'], wsp_full)[0]
    
    results_FSB[f'FSB_NoVD_bin{j}'] = FSB0
    results_FSB[f'FSB_NoVD_Fullsky_bin{j}'] = FSB0_full

    if pars.DoVD == True:
        FSB1 = fsb.get_cl(ff1['fN'], ff1[f'f{j}'], wsp_mask)[0]
        FSB_dep = fsb.get_cl(ff_dep['fN'], ff_dep[f'f{j}'], wsp_mask)[0]
        FSB_dep_full = fsb.get_cl(ff_dep_full['fN'], ff_dep_full[f'f{j}'], wsp_full)[0]
        FSB_rew = fsb.get_cl(ff_rew['fN'], ff_rew[f'f{j}'], wsp_mask)[0]

        results_FSB[f'FSB_VD_bin{j}'] = FSB1
        results_FSB[f'FSB_Deprojected_bin{j}'] = FSB_dep
        results_FSB[f'FSB_Deprojected_Fullsky_bin{j}'] = FSB_dep_full
        results_FSB[f'FSB_Reweighted_bin{j}'] = FSB_rew

df_FSB = pd.DataFrame(results_FSB)
df_FSB.to_csv(f'{path}FsbSALMO/FSB_{args.taskID}.data')
print('Saved FSB', flush=True)

import sys
from pathlib import Path

# Get the parent directory of the script
parent_dir = Path(__file__).resolve().parent.parent

# Add the parent directory to sys.path
sys.path.append(str(parent_dir))

import pymaster as nmt   ### Namaster
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import FSB as fsb
import pandas as pd
import argparse
import parameters as pars

parser = argparse.ArgumentParser()

parser.add_argument('taskID')

args = parser.parse_args()

# other parameters
b = 1  ## bias
dell_bin = pars.dell
nside = pars.nside
lmax = pars.lmax_filts

## mask
MaskSALMO = hp.ud_grade(hp.read_map(f'../maps_for_salmo/{pars.mask}'), nside)

## variable depth map
DepthMap = hp.ud_grade(hp.read_map(f'{pars.inputVDmap}'), nside)
DepthMap[np.isnan(DepthMap)] = 0
DepthMap = DepthMap -np.mean(DepthMap[MaskSALMO==1])


nbands = pars.nbands
dell = lmax // nbands
filters = fsb.get_filters(nbands, lmax)

## define bins
bins = nmt.NmtBin.from_nside_linear(nside, dell_bin)
ell_arr = bins.get_effective_ells()   ## l's corresponding to the bins
nbins = bins.get_n_bands()

## workspace
fmask = nmt.NmtField(MaskSALMO, None, spin=0)
wsp_full = nmt.NmtWorkspace()
wsp_full.compute_coupling_matrix(fmask, fmask, bins)


def CreateNumDens(galmap, mask):
    n_mean = np.sum(galmap[np.where(mask!=0)[0]]) / len(np.where(mask!=0)[0])
    n_dens_gals = galmap/n_mean
    return n_dens_gals

GalaxyDensity_type0 = CreateNumDens(hp.read_map(f'{pars.path_ws}GalMaps/GalCount_{args.taskID}_type0.fits'), MaskSALMO)-1
GalaxyDensity_type1 = CreateNumDens(hp.read_map(f'{pars.path_ws}GalMaps/GalCount_{args.taskID}_type1.fits'), MaskSALMO)-1

## indices:
## fField0: simulation without VD
## fField1: simulation with VD
## fField_deproj1: simulation with VD where the individual fields are deprojected and then squared
## fField_deproj2: simulation with VD where the fields with VD are squared and then the squared depth map is deprojected
## fField_deproj3: simulation with VD where the fields with VD are squared and then the non-squared depth map is deprojected
results_FSB = {}
fField0 = fsb.get_fields_from_map(GalaxyDensity_type0, MaskSALMO, filters, nside)
fField1 = fsb.get_fields_from_map(GalaxyDensity_type1, MaskSALMO, filters, nside)
fField_deproj1 = fsb.get_fields_from_map_wtemp(GalaxyDensity_type1, MaskSALMO, filters, nside, templates=[[DepthMap]])
fField_deproj2 = fsb.get_fields_from_map_wtemp2(GalaxyDensity_type1, MaskSALMO, filters, nside, templates=[[DepthMap]])
fField_deproj3 = fsb.get_fields_from_map_wtemp3(GalaxyDensity_type1, MaskSALMO, filters, nside, templates=[[DepthMap]])

for j in range(nbands):
    FSB0 = fsb.get_cl(fField0['fN'], fField0[f'f{j}'], wsp_full)[0] 
    FSB1 = fsb.get_cl(fField1['fN'], fField1[f'f{j}'], wsp_full)[0]
    FSB_deproj1 = fsb.get_cl(fField_deproj1['fN'], fField_deproj1[f'f{j}'], wsp_full)[0]
    FSB_deproj2 = fsb.get_cl(fField_deproj2['fN'], fField_deproj2[f'f{j}'], wsp_full)[0]
    FSB_deproj3 = fsb.get_cl(fField_deproj3['fN'], fField_deproj3[f'f{j}'], wsp_full)[0]
    results_FSB[f'FSB_NoVD_bin{j}'] = FSB0
    results_FSB[f'FSB_VD_bin{j}'] = FSB1
    results_FSB[f'FSB_Deprojected_bin{j} 1'] = FSB_deproj1
    results_FSB[f'FSB_Deprojected_bin{j} 2'] = FSB_deproj2
    results_FSB[f'FSB_Deprojected_bin{j} 3'] = FSB_deproj3

df_FSB = pd.DataFrame(results_FSB)
df_FSB.to_csv(f'{pars.path_ws}methods_deprojection/FSB_{args.taskID}.data')

## import modules
import sys
from pathlib import Path

# Get the parent directory of the script
parent_dir = Path(__file__).resolve().parent.parent

# Add the parent directory to sys.path
sys.path.append(str(parent_dir))

import healpy as hp
import numpy as np
import pymaster as nmt
import pandas as pd
import FSB as fsb
import argparse
from FSB import first_term
import parameters as pars

parser = argparse.ArgumentParser()

parser.add_argument('taskID')

args = parser.parse_args()

taskID = args.taskID

nside = pars.nside
path = pars.path_ws
deltaell = pars.dell
nbands = pars.nbands
lmax = pars.lmax_filts
dell = lmax // nbands
filters = fsb.get_filters(nbands, lmax)

b = pars.b
bins = nmt.NmtBin.from_edges(b[:-1], b[1:])

def CreateNumDens(galmap, mask):
    n_mean = np.sum(galmap[np.where(mask!=0)[0]]) / len(np.where(mask!=0)[0])
    n_dens_gals = galmap/n_mean
    return n_dens_gals

mask = np.ones(hp.nside2npix(nside))
mask_decals = hp.ud_grade(hp.read_map(f'../maps_for_salmo/{pars.mask}'), nside)


temp_list = pars.temp_list
temp_min = pars.temp_min

m_nstar = hp.ud_grade(hp.read_map(f'../templates/{temp_list[0]}'), nside)
m_completeness = hp.ud_grade(hp.read_map(f'../templates/{temp_list[1]}'), nside)
m_extinction = hp.ud_grade(hp.read_map(f'../templates/{temp_list[2]}'), nside)


templates = [[m_nstar], [m_completeness], [m_extinction]]
#templates = [[m_nstar]]

for i, temp in enumerate(templates):
    temp[0][np.isnan(temp[0])] = 0
    temp[0][temp[0] < temp_min[i]] = temp_min[i]
    temp[0] = temp[0] - np.mean(temp[0][mask_decals==1])

map = (CreateNumDens(hp.read_map(f'{path}GalMaps/GalCount_{taskID}_type0.fits'), mask_decals) - 1)*mask_decals


## calculate cls
fsys = nmt.NmtField(mask, [map], templates=templates)

results1t = {}

fsb_fsys = fsb.get_fields_from_map_wtemp(map, mask, filters, nside, templates=templates, sub_mean=False, iter=3)

fsbg_df = pd.read_csv(f'{path}FsbSALMO/mean_FSB.data')

## workspace
fmask = nmt.NmtField(mask, None, spin=0)
wsp = nmt.NmtWorkspace()
wsp.compute_coupling_matrix(fmask, fmask, bins)

fsb_db_arr = np.zeros((nbands, bins.get_n_bands()))
for j in range(nbands):
    fsbg = fsbg_df[f'FSB_NoVD_bin{j}'].to_numpy()

    t1 = first_term(fsys, fsys, np.array([[fsbg]]))[0,0]
    results1t[f'FSB first term bin {j}'] = t1
    print(f'Saved FSB bin {j} Deprojection Bias', flush=True)

pd.DataFrame(results1t).to_csv(f'{path}FSB_depbias/t1_{taskID}.data')
print('Done', flush=True)

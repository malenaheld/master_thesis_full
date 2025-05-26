import sys
from pathlib import Path

# Get the parent directory of the script
parent_dir = Path(__file__).resolve().parent.parent

# Add the parent directory to sys.path
sys.path.append(str(parent_dir))

import numpy as np
import parameters as pars
import healpy as hp
import pandas as pd
import pymaster as nmt
import argparse
import systematics as sys


parser = argparse.ArgumentParser()

parser.add_argument('taskID')

args = parser.parse_args()

taskID = args.taskID


path = pars.path_ws
nside = pars.nside
a_sys = pars.a

mask = hp.ud_grade(hp.read_map(f'../maps_for_salmo/{pars.mask}'), nside)


temp_list = pars.temp_list
temp_min = pars.temp_min
temp_max = pars.temp_max

m_nstar = hp.ud_grade(hp.read_map(f'../templates/{temp_list[0]}'), nside)
m_completeness = hp.ud_grade(hp.read_map(f'../templates/{temp_list[1]}'), nside)
m_extinction = hp.ud_grade(hp.read_map(f'../templates/{temp_list[2]}'), nside)


## define templates
templates = [[m_nstar], [m_completeness], [m_extinction]]
temps_names = ['$n_{star}$', 'completeness', 'extinction'] #, 'blank']
#templates = [[m_nstar]]
#temps_names = ['$n_{star}$']

for i, temp in enumerate(templates):
    temp[0][np.isnan(temp[0])] = 0
    temp[0][temp[0] < temp_min[i]] = temp_min[i]
    temp[0][temp[0] > temp_max[i]] = temp_max[i]
#    temp[0] = temp[0] - np.mean(temp[0][mask==1])

results = {}

GalCount1 = hp.ud_grade(hp.read_map(f'{path}GalMaps/GalCount_{taskID}_type1.fits'), nside)
m_ngal1 = GalCount1 / np.mean(GalCount1[np.where(mask==1)]) - 1

f_ngal_dep = nmt.NmtField(mask, [m_ngal1], templates=templates)
m_ngal_dep = f_ngal_dep.get_maps()[0]
m_rew = sys.reweight_map(GalCount1, mask, templates, a_sys=a_sys)
m_rew = m_rew / np.mean(m_rew[mask==1]) - 1

for i, (name, temp) in enumerate(zip(temps_names, templates)):
    temp = temp[0]
    bmin = np.min(temp[mask!=0])
    bmax = np.max(temp[mask!=0])
    nbins = 10
    bin_edges = np.linspace(bmin, bmax, nbins + 1)
#    ind_sort, bin_edges = get_binedges(temp, mask, nbins)
    ind_sort = np.arange(len(temp[mask!=0]))
    
    n_gals_map_dep, temp_vals = sys.CalculateSystematics(m_ngal_dep[np.where(mask!=0)][ind_sort], temp[np.where(mask!=0)][ind_sort], bin_edges)
    n_gals_map1,temp_vals = sys.CalculateSystematics(m_ngal1[np.where(mask!=0)][ind_sort], temp[np.where(mask!=0)][ind_sort], bin_edges)
    n_gals_rew, temp_vals = sys.CalculateSystematics(m_rew[np.where(mask!=0)][ind_sort], temp[np.where(mask!=0)][ind_sort], bin_edges)
    
    results[f'{name}'] = temp_vals
    results[f"Deprojected: {name}: ngals"] = n_gals_map_dep
    results[f"Reweighted: {name}: ngals"] = n_gals_rew
    results[f"VD: {name}: ngals"] = n_gals_map1

print('computed simulated VD relations', flush=True)

inputVDmap = hp.ud_grade(hp.read_map(f'{pars.inputVDmap}'), nside)
f_inputVDmap_dep = nmt.NmtField(mask, [inputVDmap-1], templates=templates)
inputVDmap_dep = f_inputVDmap_dep.get_maps()[0]

for temp, name in zip(templates, temps_names):
    temp = temp[0]
    bmin = np.min(temp[mask!=0])
    bmax = np.max(temp[mask!=0])
    nbins = 10
    bin_edges = np.linspace(bmin, bmax, nbins + 1)
#    ind_sort, bin_edges = get_binedges(temp, mask, nbins)
    ind_sort = np.arange(len(temp[mask!=0]))
    print(np.max(inputVDmap), flush=True)

    n_gals_input, temp_vals = sys.CalculateSystematics(inputVDmap[np.where(mask!=0)][ind_sort], temp[np.where(mask!=0)][ind_sort], bin_edges)
    n_gals_input_dep, temp_vals = sys.CalculateSystematics(inputVDmap_dep[np.where(mask!=0)][ind_sort], temp[np.where(mask!=0)][ind_sort], bin_edges)
    results[f"Input: {name}: ngals"] = n_gals_input
    results[f"Input Deprojected: {name}: ngals"] = n_gals_input_dep

print('computed input VD relations', flush=True)


pd.DataFrame(results).to_csv(f'{path}/sysrels/systematic_relations_{taskID}.data')
print('saved VD relations', flush=True)

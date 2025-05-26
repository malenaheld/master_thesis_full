import sys
from pathlib import Path

# Get the parent directory of the script
parent_dir = Path(__file__).resolve().parent.parent

# Add the parent directory to sys.path
sys.path.append(str(parent_dir))

import pymaster as nmt
import numpy as np
import healpy as hp
import parameters as pars

nside = pars.nside

mask = hp.ud_grade(hp.read_map(f'../maps/{pars.mask}'), nside)

map = np.ones_like(mask, dtype=float)

temp_list = pars.temp_list
temp_min = pars.temp_min

m_nstar = hp.ud_grade(hp.read_map(f'../maps/{temp_list[0]}'), nside)
m_completeness = hp.ud_grade(hp.read_map(f'../maps/{temp_list[1]}'), nside)
m_extinction = hp.ud_grade(hp.read_map(f'../maps/{temp_list[2]}'), nside)


templates = [[m_nstar], [m_completeness], [m_extinction]]
#templates = [[m_nstar]]

for i, temp in enumerate(templates):
    temp[0][np.isnan(temp[0])] = 0
    temp[0][temp[0] < temp_min[i]] = temp_min[i]
    temp[0] = temp[0] - np.mean(temp[0][mask==1])
    

## calculate analytical deprojection bias

## theory cls
cls_theo = np.load('cls_theory_ccl.npy')[:3*nside]

## field
fsys = nmt.NmtField(mask, [map], templates=templates)


deprojection_bias_cls = nmt.deprojection_bias(fsys, fsys, cl_guess=np.array([cls_theo]))

if len(templates)>1:
    np.save('depbias_multi.npy', deprojection_bias_cls)
else:
    np.save('depbias_single.npy', deprojection_bias_cls)

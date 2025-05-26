import sys
from pathlib import Path

# Get the parent directory of the script
parent_dir = Path(__file__).resolve().parent.parent

# Add the parent directory to sys.path
sys.path.append(str(parent_dir))

import pymaster as nmt   ### Namaster
import healpy as hp
import pandas as pd
import argparse
import parameters as pars
import os

parser = argparse.ArgumentParser()

parser.add_argument('taskID')

args = parser.parse_args()


# basic parameters of the simulation
nside = pars.nside
dell = pars.dell
path = pars.path_glass

### Mask used in SALMO simulation
mask = hp.read_map(f'../salmo-master/salmo-files/input/{pars.mask}')
mask = hp.ud_grade(mask, nside)

## define bins
if pars.bins == None:
   bins = nmt.NmtBin.from_nside_linear(nside, dell)  
else:
   bins = nmt.NmtBin.from_edges(pars.b[:-1], pars.b[1:])
ell_arr = bins.get_effective_ells()   ## l's corresponding to the bins


## calculate the cls for the projected matter shell
matter_map = hp.read_map(f'{path}MatterDensMaps/GLASS_map_{args.taskID}.fits')


field = nmt.NmtField(mask, [matter_map])
cls = nmt.compute_full_master(field, field, bins)[0]


results_Cls = {}
results_Cls['ell'] = ell_arr
results_Cls['Cls'] = cls


df_Cls = pd.DataFrame(results_Cls)
df_Cls.to_csv(f'{path}ClsGLASS/Cls_{args.taskID}.data')
print('Saved Cls from projection GLASS simulation', flush=True)
   

## calculate the cls for each matter shell

## select all files
path_files = f'{path}MatterShells/'
files_list = [f for f in os.listdir(path_files) if f'DECALS_denMap_{args.taskID}_run2_f1z' in f]


results_shells = {}
results_shells['ell'] = ell_arr
for i, file in enumerate(files_list):
    shell_map = hp.read_map(f'{path_files}{file}')
    
    ## calculate cls
    field = nmt.NmtField(mask, [shell_map])
    cls = nmt.compute_full_master(field, field, bins)[0]
    
    results_shells[f'shell {i+1}'] = cls
    
df_Cls = pd.DataFrame(results_shells)
df_Cls.to_csv(f'{path}ClsShells/Cls_{args.taskID}.data')
print('Saved Cls from the shells of GLASS simulation', flush=True)

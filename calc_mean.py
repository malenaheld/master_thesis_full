import parameters as pars
import pandas as pd
import os
import numpy as np
import pymaster as nmt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('path')
parser.add_argument('tag')
args = parser.parse_args()

## read parameters
nside = pars.nside
deltaell = pars.dell

print(args.path, flush=True)

## select all files
path_files = f'{args.path}/'
files_list = [f for f in os.listdir(path_files) if f'{args.tag}_' in f]
Nsim = len(files_list)   ## number of sims

file_ex = pd.read_csv(f'{path_files}{files_list[0]}') ## get keys from this file

results = {}  ## store mean results here
for key in file_ex.keys():
    array = np.zeros((Nsim, len(file_ex[file_ex.keys()[1]])))
    for i, file in enumerate(files_list):
        df = pd.read_csv(f'{path_files}{file}')
        array[i] = df[key].to_numpy()

    results[key] = np.nanmean(array,axis=0)   ## mean for key of all sims
    ## calculate uncertainty 
    if key == 'ell':  
        pass  ## no uncertainty for the ells
    else:
        results[f'{key} unc'] = np.nanstd(array, axis=0) / np.sqrt(Nsim)

## save results
df = pd.DataFrame(results)
df.to_csv(f'{path_files}mean_{args.tag}.data')



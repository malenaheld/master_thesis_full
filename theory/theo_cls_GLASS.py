import sys
from pathlib import Path

# Get the parent directory of the script
parent_dir = Path(__file__).resolve().parent.parent

# Add the parent directory to sys.path
sys.path.append(str(parent_dir))

import numpy as np
import camb
from cosmology import Cosmology
import parameters as pars

import glass.shells
import glass.ext.camb


# cosmology for the simulation
h = 0.7
Oc = 0.25
Ob = 0.05
ns = 0.965
A_s = 2e-09

# basic parameters of the simulation
lmax = pars.lmax
dx = pars.dx
nside = pars.nside

# set up CAMB parameters for matter angular power spectrum
pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2,
                       NonLinear=camb.model.NonLinear_both)
pars.InitPower.set_params(ns=ns)
#pars.set_matter_power(kmax=10)

# get the cosmology from CAMB
cosmo = Cosmology.from_camb(pars)

zb_cut = glass.shells.distance_grid(cosmo, 0,1, dx=dx)


# CAMB requires linear ramp for low redshifts
ws = glass.shells.tophat_windows(zb_cut, weight=glass.ext.camb.camb_tophat_weight)

# compute angular matter power spectra with CAMB
cls = np.array(glass.ext.camb.matter_cls(pars, lmax, ws))

## save computed power spectrum, so it does not have to be computed each time
np.save(f'cls_dx{dx}.npy', cls)


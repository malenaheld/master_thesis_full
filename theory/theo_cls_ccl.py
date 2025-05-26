import sys
from pathlib import Path

# Get the parent directory of the script
parent_dir = Path(__file__).resolve().parent.parent

# Add the parent directory to sys.path
sys.path.append(str(parent_dir))

import numpy as np
import pyccl as ccl
import parameters as pars

nside = pars.nside

# cosmology for the simulation
h = 0.7
Oc = 0.25
Ob = 0.05
ns = 0.965
A_s = 2e-09
b = 1
lmax = 3*nside

zb = np.load('../data_DELS/z.npy')

# redshift distribution
nz = np.load('../data_DELS/nz.npy')   ## weights according to the redshift distribution

cosmo = ccl.Cosmology(Omega_c = Oc, Omega_b = Ob, h = h, A_s = A_s, n_s = ns, matter_power_spectrum='camb', m_nu=.06, extra_parameters = {"camb": {"halofit_version": "mead2020", "HMCode_A_baryon": 3.13, "HMCode_eta_baryon": 0.603, "HMCode_logT_AGN": 7.8, "lmax": lmax, 'dark_energy_model':"fluid"}}, baryonic_effects=None, mass_split='single')

### calculate angular power spectrum
gals = ccl.NumberCountsTracer(cosmo, dndz = (zb, nz), bias=(zb, b*np.ones_like(zb)), has_rsd=False)

ells = np.arange(3*nside)
cls_theo = ccl.angular_cl(cosmo, gals, gals, ells, l_limber=1000)

np.save('cls_theory_ccl.npy', cls_theo)

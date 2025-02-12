# -*- coding: utf-8 -*-
"""
Library of functions for scuffem

Created on Thu Oct 10 14:12 2024

@author: PanxoPanza
"""

import numpy as np
from . import nklib as nk

def make_scuff_runfiles(lam, Material=None):
    """
    Create OmegaList and dielectric properties files for scuff-EM simulations.
    
    Input:
        lam: Wavelength range (um)
        Material: dictionary with:
            keys: materials name for .dat file
            values: nk data in ndarray (dtype=complex)
    """
    # Constants
    c0 = 299792458  # speed of light in m/s
    # w = 2 * np.pi * c0 / lam * 1E6  # angular frequency

    if Material is None:
        Material = {}
    elif not isinstance(Material, dict):
        raise ValueError('Material variable must be a dictionary')

    # Create OmegaList file
    with open('OmegaList.dat', 'w') as f:
        for iw in range(len(lam)):
            f.write(f"{2 * np.pi / lam[iw]:.6f}")
            if iw < len(lam) - 1:
                f.write('\n')

    # Create frequency range for .dat files
    lambda_mat = np.insert(lam, 0, min(lam) * 0.9)
    lambda_mat = np.append(lam, max(lam) * 1.1)
    
    w = 2 * np.pi * c0 / lambda_mat * 1E6

    # export *.dat files
    if Material:
        for mat_label in Material.keys():
            nk_raw = Material[mat_label]

            if not isinstance(nk_raw, np.ndarray):
                raise ValueError('"%s" values are not ndarray' % mat_label)
            
            if len(lam) != len(nk_raw):
                raise ValueError('size of "%s" and "lam" arrays must be equal' % mat_label)

            # interpolate original nk raw data using "lambda_mat" range
            nk_data = np.interp(lambda_mat, lam, nk_raw.astype(complex))

            eps = nk_data**2 # get dielectric constant

            # export to a .dat file
            with open(f"{mat_label}.dat", 'w') as f:
                for wi, epsilon in zip(w, eps):
                    f.write(f"{wi:.6e} {epsilon.real:.5e}+{epsilon.imag:.5e}i\n")



# -*- coding: utf-8 -*-
"""
Library of functions for scuffem

Created on Thu Oct 10 14:12 2024

@author: PanxoPanza
"""

import numpy as np
from . import nklib as nk

def make_scuff_runfiles(lambda_range, Material=None):
    """
    Create OmegaList and dielectric properties files for scuff-EM simulations.
    
    Input:
        lambda_range: Wavelength range (um)
        Material: name of material (list of strings)
    """
    # Constants
    c0 = 299792458  # speed of light in m/s
    w = 2 * np.pi * c0 / lambda_range * 1E6  # angular frequency

    if Material is None:
        Material = []
    if isinstance(Material, str):
        Material = [Material]  # convert to list if a single string is provided

    # Create OmegaList file
    with open('OmegaList.dat', 'w') as f:
        for iw in range(len(lambda_range)):
            f.write(f"{2 * np.pi / lambda_range[iw]:.6f}")
            if iw < len(lambda_range) - 1:
                f.write('\n')

    # Create optical properties files
    lambda_mat = np.linspace(min(lambda_range) * 0.9, max(lambda_range) * 1.1, 200)
    w = 2 * np.pi * c0 / lambda_mat * 1E6

    if Material:
        for mat in Material:
            # Assuming `eps_mat` functions are available (e.g., `eps_gold`, `eps_silver`, etc.)
            nk_data = eval(f"nk.{mat}(lambda_mat)") 
            eps = nk_data**2
            with open(f"{mat}.dat", 'w') as f:
                for wi, epsilon in zip(w, eps):
                    f.write(f"{wi:.6e} {epsilon.real:.4f}+{epsilon.imag:.4f}i\n")



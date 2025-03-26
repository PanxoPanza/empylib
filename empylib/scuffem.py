# -*- coding: utf-8 -*-
"""
Library of functions for scuffem

Created on Thu Oct 10 14:12 2024

@author: PanxoPanza
"""

import numpy as np
import pandas as pd
from . import nklib as nk

def make_spectral_files(lam, Material=None):
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
def read_scatter_PFT(FileName):
    # Extraemos la info en un dataframe
    df = pd.read_csv(FileName,comment = '#', sep='\s+', header = None, index_col = 0)

    # Asignar nombres a las columnas
    df.columns = ['Label', 'Pabs', 'Psca', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']

    # Establecer "omega" como índice
    df.index.name = 'Omega'

    # Identificar todas las etiquetas únicas en "surface"
    unique_labels = df["Label"].unique()

    # Crear un diccionario para guardar los DataFrames por etiqueta
    objectID = {}

    # Para cada etiqueta, crear un DataFrame con los datos correspondientes (excluyendo la columna 'surface')
    for label in unique_labels:
        objectID[label] = df[df["Label"] == label].drop(columns="Label")

    return objectID

def read_avescatter(FileName):
    # Extraemos la info en un dataframe
    df = pd.read_csv(FileName,comment = '#', sep='\s+', header = None, index_col = 1)

    # Eliminamos la primera columna
    df.drop([0], axis=1, inplace=True)

    # Establecer "omega" como índice
    df.index.name = 'Omega'

    # Asignar nombres a las columnas
    df.columns = ['Label', '<Cabs>', '<Csca>', '<Cpr>']

    # Identificar todas las etiquetas únicas en "surface"
    unique_labels = df["Label"].unique()

    # Crear un diccionario para guardar los DataFrames por etiqueta
    objectID = {}

    # Para cada etiqueta, crear un DataFrame con los datos correspondientes (excluyendo la columna 'surface')
    for label in unique_labels:
        objectID[label] = df[df["Label"] == label].drop(columns="Label")

    return objectID


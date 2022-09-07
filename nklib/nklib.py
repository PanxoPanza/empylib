# -*- coding: utf-8 -*-
"""
Library of tabulated refractive index

Created on Sun Nov  7 17:25:53 2021

@author: PanxoPanza
"""
import os
import platform
import numpy as np 
from scipy.interpolate import CubicSpline
from warnings import warn

def get_nkfile(lam, MaterialName):
    '''
    Reads an *.nk file and returns an interpolated
    1D numpy array with the complex refractive index
    of the material Hola chuquitin
    
    Parameters
    ----------
    lam : ndarray
        Wavelengths to interpolate (um).
    MaterialName : string
        Name of *.nk file

    Returns
    -------
    N : ndarray
        Interpolated complex refractive index
    data: ndarray
        Original tabulated data from file
    '''
    
    # retrieve local path
    dir_separator = '\\' # default value
    if platform.system() == "Linux":    # linux
        dir_separator= '/'

    elif platform.system() == 'Darwin': # OS X
        dir_separator='/'

    elif platform.system() == "Windows":  # Windows...
        dir_separator='\\'

    dir_path = os.path.dirname(__file__) + dir_separator
    filename = dir_path + MaterialName + '.nk'
   
    # check if file exist
    assert os.path.isfile(filename), 'File not found'
    
    data = np.genfromtxt(filename)
    assert data.shape[1] <= 3, 'wrong file format'

    # create complex refractive index using interpolation form nkfile
    n = CubicSpline(data[:,0],data[:,1],extrapolate=False)
    k = CubicSpline(data[:,0],data[:,2],extrapolate=False)
    
    N = n(lam) + 1j*k(lam)
    
    # Add a flat nk for extrapolated values (warn user)
    if lam[ 0] < data[ 0,0] :
        warn('Extrapolating from %.3f to %.3f' % (lam[0], data[0,0]))
        N[lam <= data[ 0,0]] = data[ 0,1] + 1j*data[ 0,2]
        
    if lam[-1] > data[-1,0] :
        warn('Extrapolating from %.3f to %.3f' % (data[-1,0], lam[-1]))
        N[lam >= data[-1,0]] = data[-1,1] + 1j*data[-1,2]
    
    return N, data

'''
    --------------------------------------------------------------------
                    dielectric constant models
    --------------------------------------------------------------------
'''
def lorentz(epsinf,wp,wn,gamma,lam):
    '''
    Refractive index from Lorentz model

    Parameters
    ----------
    epsinf : float
        dielectric constant at infinity.
    wp : float
        Plasma frequency, in eV (wp^2 = Nq^2/eps0 m).
    wn : float
        Natural frequency in eV
    gamma : float
        Decay rate in eV
    lam : linear np.array
        wavelength spectrum in um

    Returns
    -------
    complex refractive index

    '''
    # define constants
    eV = 1.602176634E-19          # eV to J (conversion)
    hbar = 1.0545718E-34          # J*s (plank's constan)
    
    
    w = 2*np.pi*3E14/lam*hbar/eV  # conver from um to eV 
    
    return np.sqrt(epsinf + wp**2/(wn**2 - w**2 - 1j*gamma*w))


def drude(epsinf,wp,gamma,lam):
    '''
    Refractive index from Drude model

    Parameters
    ----------
    epsinf : float
        dielectric constant at infinity.
    wp : float
        Plasma frequency, in eV (wp^2 = Nq^2/eps0 m).
    gamma : float
        Decay rate in eV
    lam : linear np.array
        wavelength spectrum in um

    Returns
    -------
    complex refractive index

    '''
    # define constants
    eV = 1.602176634E-19          # eV to J (conversion)
    hbar = 1.0545718E-34          # J*s (plank's constan)
    
    
    w = 2*np.pi*3E14/lam*hbar/eV  # conver from um to eV 
    
    return np.sqrt(epsinf - wp**2/(w**2 + 1j*gamma*w))

'''
    --------------------------------------------------------------------
                            Target functions
    --------------------------------------------------------------------
'''

# refractive index of SiO2
SiO2 = lambda lam: get_nkfile(lam, 'sio2_Palik_Lemarchand2013')[0]

# refractive index of TiO2
TiO2 = lambda lam: get_nkfile(lam, 'tio2_Siefke2015')[0]

# refractive index of Gold
gold = lambda lam: get_nkfile(lam, 'au_Olmon2012_evap')[0]

# refractive index of Copper
Cu   = lambda lam: get_nkfile(lam, 'cu_Babar2015')[0]

# refractive index of Aluminium
Al   = lambda lam: get_nkfile(lam, 'al_Rakic1995')[0]

# refractive index of Silver
silver = lambda lam: get_nkfile(lam, 'ag_Ciesielski2017')[0]

# refractive index of Silicon
Si   = lambda lam: get_nkfile(lam, 'si_Schinke2017')[0]

# refractive index of water
H2O  = lambda lam: get_nkfile(lam, 'h2o_Hale1973')[0]
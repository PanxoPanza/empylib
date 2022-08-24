# -*- coding: utf-8 -*-
"""
Library of tabulated refractive index

Created on Sun Nov  7 17:25:53 2021

@author: PanxoPanza
"""
import os
import platform
import numpy as np 

def get_nkfile(lam, MaterialName):
    '''
    Reads an *.nk file and returns an interpolated
    1D numpy array with the complex refractive index
    of the material Hola chuquitin
    
    Parameters
    ----------
    lam : 1D numpy array
        Wavelengths to interpolate (um).
    MaterialName : string
        Name of *.nk file

    Returns
    -------
    1D numpy array
        Interpolated complex refractive index
    3D numpy array (optional)
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

    # if lam is scalar length = 1, else get arrays length
    if np.isscalar(lam):
        len_lam = 1
    else:
        len_lam = len(lam)
    
    # create complex refractive index using interpolation form nkfile
    N = np.zeros((len_lam,2))
    for i in range(1,data.shape[1]):
        # interpolate the refractive index according to "lam"
        N[:,i-1] = np.interp(lam,data[:,0],data[:,i])
        
        # for extrapolated values make out = 0
        N[:,i-1] = N[:,i-1]*(lam<=np.max(data[:,0]))*(lam>=np.min(data[:,0])) 

    return N[:,0] + 1j*N[:,1] , data

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

# refractive index of sio2
sio2 = lambda lam: get_nkfile(lam, 'sio2_Palik_Lemarchand2013')[0]

# refractive index of tio2
tio2 = lambda lam: get_nkfile(lam, 'tio2_Siefke2016')[0]

# refractive index of gold
gold = lambda lam: get_nkfile(lam, 'au_Olmon2012_evap')[0]

# refractive index of silicon
si = lambda lam: get_nkfile(lam, 'si_Schinke2017')[0]

# refractive index of water
h2o = lambda lam: get_nkfile(lam, 'h2o_Hale1973')[0]
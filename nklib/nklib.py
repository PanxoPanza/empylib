# -*- coding: utf-8 -*-
"""
Library of tabulated refractive index

Created on Sun Nov  7 17:25:53 2021

@author: PanxoPanza
"""
import os
import platform
import numpy as np 
from scipy.interpolate import interp1d
from warnings import warn

def get_nkfile(lam, MaterialName):
    '''
    Reads an *.nk file and returns an interpolated
    1D numpy array with the complex refractive index
    of the material
    
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
    # convert lambda to list
    if np.isscalar(lam): lam = np.array([lam,])    
    
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
    n = interp1d(data[:,0],data[:,1],bounds_error=False)
    k = interp1d(data[:,0],data[:,2],bounds_error=False)
    
    N = n(lam) + 1j*k(lam)*(k(lam)>=0)
    
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

def emt_brugg(f1,nk_1,nk_2):
    '''
    Effective permitivity based on Bruggersman theory
    
        Parameters
    ----------
    nk_1: ndarray
        refractive index of inclussions
    
    nk_2: ndarray
        refractive index of host
    
    f1: float   
        filling fraction of material eps1

    Returns
    -------
    complex refractive index of effective media
    '''
    # check simple cases first
    if f1 == 0:     # no inclusions
        return nk_2
    elif f1 == 1:   # no host
        return nk_1

    # prepare variables
    f2 = 1 - f1
    eps_1, eps_2 = nk_1**2, nk_2**2 # convert refractive index to dielectric constants
    
    # if eps_1 and eps_2 are scalar convert both to 1D ndarray
    if np.isscalar(eps_1) and np.isscalar(eps_2):
        eps_1 = np.array([eps_1,])
        eps_2 = np.array([eps_2,])
    
    # else, scale the scalar variable with the ndarray. 
    else: 
        # eps_1 is scalar, create a constant array of len(eps_2)
        if   np.isscalar(eps_1):
            eps_1 = eps_1*np.ones(len(eps_2))
        
        # eps_2 is scalar, create a constant array of len(eps_1)
        elif np.isscalar(eps_2):
            eps_2 = eps_2*np.ones(len(eps_1))
        
        # if both are ndarrays, assert they are both equal length
        else:
            assert len(eps_1) == len(eps_2), 'size of eps_1 and eps_2 must be equal'

    # compute effective dielectric constant ussing Bruggerman theory.
    eps_m = 1/4.*((3*f1 - 1)*eps_1 + (3*f2 - 1)*eps_2                           \
            - np.sqrt(((3*f1 - 1)*eps_1 + (3*f2 - 1)*eps_2)**2 + 8*eps_1*eps_2))
    
    for i in range(len(eps_m)):
        if eps_m[i].imag < 0  or (eps_m[i].imag < 1E-10 and eps_m[i].real < 0):
            eps_m[i] =  eps_m[i] + \
                1/2*np.sqrt(((3*f1 - 1)*eps_1[i] + (3*f2 - 1)*eps_2[i])**2 \
                + 8*eps_1[i]*eps_2[i]) 
    
    # if eps_1 and eps_2 were scalar, return a single scalar value
    if len(eps_m) == 1: return np.sqrt(eps_m[0])
    else :              return np.sqrt(eps_m)
'''
    --------------------------------------------------------------------
                            Target functions
    --------------------------------------------------------------------
'''

#------------------------------------------------------------------------------
#                                   Oxides
# refractive index of SiO2
SiO2 = lambda lam: get_nkfile(lam, 'sio2_Palik_Lemarchand2013')[0]

# refractive index of TiO2
TiO2 = lambda lam: get_nkfile(lam, 'tio2_Siefke2015')[0]

# refractive index of amorphous GeSbTe (GST)
aGST = lambda lam: get_nkfile(lam, 'aGST_Du2016')[0]

# refractive index of crystaline GeSbTe (GST)
cGST = lambda lam: get_nkfile(lam, 'cGST_Du2016')[0]

#------------------------------------------------------------------------------
#                                   Inorganics
# refractive index of Silicon
Si   = lambda lam: get_nkfile(lam, 'si_Schinke2017')[0]

#------------------------------------------------------------------------------
#                                   Metals
# refractive index of Gold
gold = lambda lam: get_nkfile(lam, 'au_Olmon2012_evap')[0]

# refractive index of Silver
silver = lambda lam: get_nkfile(lam, 'ag_Ciesielski2017')[0]

# refractive index of Copper
Cu   = lambda lam: get_nkfile(lam, 'cu_Babar2015')[0]

# refractive index of Aluminium
Al   = lambda lam: get_nkfile(lam, 'al_Rakic1995')[0]

#------------------------------------------------------------------------------
#                                   Polymers
# refractive index of HDPE
HDPE  = lambda lam: get_nkfile(lam, 'HDPE_Palik')[0]

# refractive index of HDPE
PDMS  = lambda lam: get_nkfile(lam, 'PDMS_Zhang2020_Querry1987')[0]

# refractive index of PMMA
PMMA  = lambda lam: get_nkfile(lam, 'PMMA_Zhang2020')[0]

# refractive index of PVDF-HFP
PVDF  = lambda lam: get_nkfile(lam, 'PVDF-HFP_Mandal2018')[0]

#------------------------------------------------------------------------------
#                                   Others
# refractive index of water
H2O  = lambda lam: get_nkfile(lam, 'h2o_Hale1973')[0]
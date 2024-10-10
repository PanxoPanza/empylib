# -*- coding: utf-8 -*-
"""
This library contains reference spectra:
    AM1.5
    Plank's distribution
    Atmospheric Transmittance

Created on Fri Jan 21 16:05:48 2022

@author: PanxoPanza
"""

import os
import platform
import numpy as np 
import empylib as em

def read_spectrafile(lam, MaterialName, get_from_local_path = False):
    '''
    Reads a textfile and returns an interpolated
    1D numpy array with the complex refractive index
    of the material
    
    Parameters
    ----------
    lam : 1D numpy array
        Wavelengths to interpolate (um).
    MaterialName : string
        Name of textfile (with extension)

    Returns
    -------
    tupple with
        1.Interpolated complex refractive index
        2.Tabulated data used for interpolation (optional)
    '''
    from .utils import _ndarray_check
    from pathlib import Path

    # check if lam is not ndarray
    lam, lam_isfloat = _ndarray_check(lam)   
    
    # retrieve local path
    if get_from_local_path:
        # if function is called locally
        caller_directory = Path(__file__).parent / 'spectra_data'
    else :
        # if function is called from working directory (where the function is called)
        caller_directory = Path.cwd()

    # Construct the full path of the file
    file_path = caller_directory / MaterialName   
   
    # check if file exist
    assert file_path.exists(), 'File not found'
    
    # check number of columns in file
    data = np.genfromtxt(file_path)
    assert data.shape[1] <= 2, 'wrong file format'
    
    # run interpolation based on lam
    if lam_isfloat:
        len_lam = 1
    else:
        len_lam = len(lam)
    
    out = np.zeros((len_lam,1))
    out = np.interp(lam,data[:,0],data[:,1])
    
    # for extrapolated values make out = 0
    out = out*(lam<=np.max(data[:,0]))*(lam>=np.min(data[:,0])) 

    return out, data

def AM15(lam,spectra_type='global'):
    '''
    AM1.5 spectra

    Parameters
    ----------
    lam : 1D float array (or scalar)
        wavelength in um.

    Returns
    -------
    Interpolated AM1.5 spectra

    '''
    # interpolate values according to lam spectra
    lam = lam*1E3 # change units to nm
    
    if spectra_type == 'global':
        Isun = read_spectrafile(lam,'AM15_Global.txt', True)[0]
    elif spectra_type == 'direct':
        Isun = read_spectrafile(lam,'AM15_Direct.txt', True)[0]
    
    # keep only positive values
    if not np.isscalar(Isun):
       Isun[Isun<0]=0
    
    return Isun*1E3  # spectra in W/m2 um

def T_atmosphere(lam):
    '''
    IR Transmissivity of atmosphere taken from:
        IR Transmission Spectra, Gemini Observatory Kernel Description. 
        http://www.gemini.edu/?q/node/10789, accessed Sep 27, 2018.

    Parameters
    ----------
    lam : 1D float array (or scalar)
        wavelength in um.

    Returns
    -------
    Interpolated Transmissivity of the atmosphere

    '''
    # interpolate values according to lam spectra
    T_atm =  read_spectrafile(lam,'T_atmosphere.txt', True)[0]
    
    # keep only positive values
    if not np.isscalar(T_atm):
        T_atm[T_atm<0]=0
    
    return T_atm
    
def Bplanck(lam,T, unit = 'wavelength'):
    '''
    Spectral Plank's black-body distribution

    Parameters
    ----------
    lam : narray
        wavelength (um).
        
    T : float value
        Temperature of the distribution (K).
        
    unit string, optional
        units for integration. Options are:
            "wavelength" spectral irradiance in wavelength units (microns)
            "frequency" spectral irradiance in frequency units (hertz)

    Returns
    -------
    Ibb: ndarray
         Spectral irradiance in W/m^2-um-sr (wavelength) W/m^2-Hz-sr (frequency)
    '''
    
    # define constants
    c0 = em.speed_of_light     # m/s (speed of light)
    hbar = em.hbar          # eV*s (reduced planks constant)
    h = 2*np.pi*hbar           # J*s (planks constant)
    kB = em.kBoltzmann      # eV/K (Boltzmann constant)
    
    #-------------------------------------------------------------------------
    # Plank distribution in W/m^2-m-sr
    #-------------------------------------------------------------------------
    if unit == 'wavelength': 
        ll = lam*1E-6 # change wavelength units to m
        
        # compute planks distribbution
        Ibb = 2*h*c0**2./ll**5*1/(np.exp(h*c0/(ll*T*kB)) - 1)*1E-6
    
    #-------------------------------------------------------------------------
    # Plank distribution in W/m^2-Hz-sr
    #-------------------------------------------------------------------------
    elif unit == 'frequency':
        vv = c0/lam*1E6      # convert wavelength to frequency (Hz)
        
        # compute planks distribbution
        Ibb = 2*h*vv**3/c0**2*          \
              1/(np.exp(h*vv/(kB*T)) - 1)
    
    return Ibb

def yCIE_lum(lam):
    '''
    CIE photoscopic luminosity function from Stockman & Sharpe as a function of wavelength

    Parameters
    ----------
    lam : 1D float array (or scalar)
        wavelength in um.

    Returns
    -------
    Interpolated CIE lum

    '''
    # interpolate values according to lam spectra
    lam = lam*1E3 # change units to nm
    
    yCIE = read_spectrafile(lam,'CIE_lum.txt', True)[0]
    
    # keep only positive values
    if not np.isscalar(yCIE):
       yCIE[yCIE<0]=0
    
    return yCIE
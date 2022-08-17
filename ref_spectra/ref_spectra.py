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
import numpy as np 
import empylib as em

def read_spectrafile(lam, MaterialName):
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
    
    dir_path = os.path.dirname(__file__) + '\\'
    filename = dir_path + MaterialName
   
    # check if file exist
    assert os.path.isfile(filename), 'File not found'
    
    # check number of columns in file
    data = np.genfromtxt(filename)
    assert data.shape[1] <= 2, 'wrong file format'
    
    # run interpolation based on lam
    if np.isscalar(lam):
        len_lam = 1
    else:
        len_lam = len(lam)
    
    out = np.zeros((len_lam,1))
    out = np.interp(lam,data[:,0],data[:,1])
    
    # for extrapolated values make out = 0
    out = out*(lam<=np.max(data[:,0]))*(lam>=np.min(data[:,0])) 

    return out, data

def AM15(lam):
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
    Isun =  read_spectrafile(lam,'AM15.txt')[0]
    
    # keep only positive values
    if not np.isscalar(Isun):
       Isun[Isun<0]=0
    
    return Isun

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
    T_atm =  read_spectrafile(lam,'T_atmosphere.txt')[0]
    
    # keep only positive values
    if not np.isscalar(T_atm):
        T_atm[T_atm<0]=0
    
    return T_atm
    
def Plank_BB(lam,T, unit = 'wavelength'):
    '''
    Spectral Plank's black-body distribution

    Parameters
    ----------
    lam : 1D float array (or scalar)s
        wavelength in um.
    T : float value
        Temperature of the distribution.

    Returns
    -------
    Ibb: 1D float array (or scalar)
         Spectral irradiance in W/m^2-m (wavelength) W/m^2-Hz (frequency)
    '''
    
    # define constants
    c0 = em.speed_of_light  # m/s (speed of light)
    hbar = em.hbar          # J*s (reduced planks constant)
    h = 2*np.pi*hbar        # J*s (planks constant)
    kB = em.kBoltzmann      # J/K (Boltzmann constant)
    
    #-------------------------------------------------------------------------
    # Plank distribution in W/m^2-m
    #-------------------------------------------------------------------------
    if unit == 'wavelength': 
        l = lam*1E-6 # change wavelength units to m
        
        # ll and TT are dim(T)xdim(l) arrays. ll runs in the column
        ll, TT = np.meshgrid(l,T)
        
        # compute planks distribbution
        Ibb = 2*h*c0**2./ll**5*1/(np.exp(h*c0/(ll*TT*kB)) - 1)
    
    #-------------------------------------------------------------------------
    # Plank distribution in W/m^2-Hz
    #-------------------------------------------------------------------------
    elif unit == 'frequency':
        v = c0/lam*1E6      # convert wavelength to frequency (Hz)
        
        # vv and TT are dim(T)xdim(l) arrays. vv runs in the column
        vv, TT = np.meshgrid(v,T)
        
        # compute planks distribbution
        Ibb = 2*h*vv**3/c0**2*          \
              1/(np.exp(h*vv/(kB*TT)) - 1)
    
    return Ibb
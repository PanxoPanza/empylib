# -*- coding: utf-8 -*-
"""
This library contains reference spectra:
    AM1.5
    Plank's distribution
    Atmospheric Transmittance

Created on Fri Jan 21 16:05:48 2022

@author: PanxoPanza
"""

import numpy as _np 
import empylib as _em
from .utils import _ndarray_check, _local_to_global_angles
from pathlib import Path as _Path

# Global cache for loaded files and interpolators
_file_cache = {}

def read_spectrafile(lam, MaterialName, get_from_local_path=False, return_data=False):
    """
    Reads a text file and returns an interpolated 1D NumPy array 
    for the specified material's spectral data.

    Parameters
    ----------
    lam : float or ndarray
        Wavelengths (in µm) to interpolate.
    MaterialName : str
        Name of the file (with extension).
    get_from_local_path : bool, optional
        If True, reads from the script's folder instead of the working directory.
    return_data : bool, optional
        If True, returns both the interpolated values and the full data array.

    Returns
    -------
    out : ndarray
        Interpolated values at requested wavelengths.
    data : ndarray (optional)
        Original tabulated data.
    """

    lam = _np.atleast_1d(lam)

    # Resolve path
    if get_from_local_path:
        caller_directory = _Path(__file__).parent / 'spectra_data'
    else:
        caller_directory = _Path.cwd()

    file_path = str(caller_directory / MaterialName)

    # Check and load file from cache or disk
    if file_path not in _file_cache:
        # Load file
        assert _Path(file_path).exists(), f"File not found: {file_path}"
        data = _np.genfromtxt(file_path)

        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("Invalid file format; expected two columns.")

        # Save to cache
        _file_cache[file_path] = data
    else:
        data = _file_cache[file_path]

    # Interpolate
    out = _np.interp(lam, data[:, 0], data[:, 1], left=0, right=0)

    if return_data:
        return out, data
    return out

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
        Isun = read_spectrafile(lam,'AM15_Global.txt', True)
    elif spectra_type == 'direct':
        Isun = read_spectrafile(lam,'AM15_Direct.txt', True)
    
    # keep only positive values
    Isun = _np.clip(Isun, 0, None)
    
    return Isun*1E3  # spectra in W/m2 um

def T_atmosphere(lam):
    '''
    Spectral transmissivity of the atmosphere for an horizontal surface 
    at normal incidence. Data taken from:
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
    T_atm =  read_spectrafile(lam,'T_atmosphere.txt', True)
    
    # keep only positive values
    T_atm = _np.clip(T_atm, 0, None)

    return T_atm

def T_atmosphere_hemi(lam, beta_tilt=0):
    """
    Computes the hemispherical atmospheric transmittance spectrum over a surface tilted at a given angle.
    This function integrates the directional atmospheric transmittance over a hemisphere centered 
    on a surface tilted by `beta_tilt` degrees. It accounts for the angular dependence of 
    radiative path length through the atmosphere, assuming unity transmittance at grazing angles 
    (zenith > 90°).

    Parameters
    ----------
    lam : 1D ndarray
        Wavelengths in micrometers [μm].
    beta_tilt : float, optional
        Tilt angle of the surface in degrees with respect to the vertical (default is 0°).

    Returns
    -------
    T_hemi : 1D ndarray
        Hemispherical atmospheric transmittance spectrum corresponding to each wavelength in `lam`.

    Notes
    -----
    - The integration is performed over solid angles using a weighted cosine projection.
    - Transmittance is assumed to be 1 for zenith angles greater than 90°, consistent with
      complete atmospheric opacity at grazing incidence.
    - The output is shifted to ensure minimum transmittance starts from 0 for normalization purposes.
    """

    beta = _np.radians(beta_tilt)

    # Angular grid
    theta_i = _np.linspace(0, _np.pi, 30)
    phi_i = _np.linspace(0, 2 * _np.pi, 30)
    tt, pp = _np.meshgrid(theta_i, phi_i, indexing='ij')  # shape: (T, P)

    theta, phi = _local_to_global_angles(tt, pp, beta, phi_tilt=0)  # shape: (T, P)

    # Compute angular weights
    weight = _np.cos(tt) * _np.sin(tt)  # shape: (T, P)

    # Flatten angles
    theta_flat = theta.ravel()  # shape: (N,)
    weight_flat = weight.ravel()  # shape: (N,)

    # Transmission mask
    mask = theta_flat < _np.pi / 2
    cos_theta = _np.cos(theta_flat[mask])  # shape: (M,)

    # T_atmosphere for all wavelengths
    T_vec = T_atmosphere(lam)[:, None]  # shape: (L,1)

    # Compute directional emissivity: (L, N)
    trans = _np.ones((len(lam), len(theta_flat)))  # shape: (L, N)
    trans[:, mask] = T_vec**(1 / cos_theta)  # broadcasting over wavelengths

    # Integrate over angles
    integrand = trans * weight_flat  # shape: (L, N)
    dphi = phi_i[1] - phi_i[0]
    dtheta = theta_i[1] - theta_i[0]

    # Sum over all angles
    T_hemi = _np.sum(integrand, axis=1) * dphi * dtheta / _np.pi # shape: (L,)

    # Adjust to enssure 0 < T < 1
    T_hemi = T_hemi - _np.min(T_hemi)
    T_hemi = T_hemi/max(_np.max(T_hemi), 1.0)

    return T_hemi
    
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
    c0 = _em.speed_of_light     # m/s (speed of light)
    hbar = _em.hbar          # eV*s (reduced planks constant)
    h = 2*_np.pi*hbar           # J*s (planks constant)
    kB = _em.kBoltzmann      # eV/K (Boltzmann constant)
    
    #-------------------------------------------------------------------------
    # Plank distribution in W/m^2-m-sr
    #-------------------------------------------------------------------------
    if unit == 'wavelength': 
        ll = lam*1E-6 # change wavelength units to m
        
        # compute planks distribbution
        Ibb = 2*h*c0**2./ll**5*1/(_np.exp(h*c0/(ll*T*kB)) - 1)*1E-6
    
    #-------------------------------------------------------------------------
    # Plank distribution in W/m^2-Hz-sr
    #-------------------------------------------------------------------------
    elif unit == 'frequency':
        vv = c0/lam*1E6      # convert wavelength to frequency (Hz)
        
        # compute planks distribbution
        Ibb = 2*h*vv**3/c0**2*          \
              1/(_np.exp(h*vv/(kB*T)) - 1)
    
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
    
    yCIE = read_spectrafile(lam,'CIE_lum.txt', True)
    
    # keep only positive values
    yCIE = _np.clip(yCIE, 0, None)
    
    return yCIE
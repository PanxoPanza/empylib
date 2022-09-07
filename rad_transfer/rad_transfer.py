# -*- coding: utf-8 -*-
"""
Library of radiative transfer function

Created on Sun Nov  7 17:25:53 2021

@author: PanxoPanza
"""
import os
import sys

empylib_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0,empylib_folder)

import numpy as np
from numpy import meshgrid, cos, sin, sqrt, conj, real, abs, pi, exp
import miescattering as mie
import waveoptics as wv

def T_beer_lambert(lam,theta, tfilm, N,fv,D,Np):
    '''
    Transmittance from Beer-Lamberts law for a film with spherical particles

    Parameters
    ----------
    lam : ndaray
        Wavelength range in microns.
        
    theta : float
        Angle of incidence in deg.
        
    tfilm : float
        Film Thickness in microns.
        
    N : tuple
        Refractive index above, in, and below the film. Length of 
        N must be 3.
        
    fv : float
        Particle's volume fraction in %.
        
    D : float
        Particle's diameter in microns
    
    Np : ndarray or float
        Refractive index of particles. If ndarray, the size must be equal to
        len(lam)

    Returns
    -------
    Tspec : ndarray
        Spectral specular transmisivity
        
    Ttot : ndarray
        Spectral total transmisivity

    '''
    if type(lam) is float : lam = np.array([lam]) # convert lam to ndarray
    assert len(N) == 3, 'length of N must be == 3'
    if not type(Np) is float:
        assert len(Np) == len(lam), 'Np must be either a float or an ndarray of size len(lam)'
    
    Tp = wv.incoh_multilayer(lam, theta, N, tfilm, pol = 'TM')[1]
    Ts = wv.incoh_multilayer(lam, theta, N, tfilm, pol = 'TE')[1]
    T = (Ts + Tp)/2
    
    # Get extinction and scattering efficiency of the sphere
    qext, qsca = mie.scatter_efficiency(lam, N[1], Np, D)[:2]
    
    qabs = qext - qsca # absorption efficiency
    
    Ac = pi*D**2/4 # cross section area of sphere
    Vp = pi*D**3/6 # volume of sphere
    cabs = Ac*qabs # absorption cross section
    cext = Ac*qext # extinction cross section
    
    Ttot = T*exp(-fv/Vp*cabs*tfilm)
    Tspec = T*exp(-fv/Vp*cext*tfilm)
    
    return Ttot, Tspec
# -*- coding: utf-8 -*-
"""
Library of radiative transfer function

Created on Sun Nov  7 17:25:53 2021

@author: PanxoPanza
"""
import os
import sys

# empylib_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# sys.path.insert(0,empylib_folder)

import numpy as np
from numpy import meshgrid, cos, sin, sqrt, conj, real, abs, pi, exp
from . import miescattering as mie
from . import waveoptics as wv
import iadpython as iad

def T_beer_lambert(lam,theta, tfilm, Nlayer,fv,D,Np):
    '''
    Transmittance and reflectance from Beer-Lamberts law for a film with 
    spherical particles. Reflectance is cokmputed from classical formulas for 
    incoherent light incident on a slab between two semi-infinite media 
    (no scattering is considered for this parameter)

    Parameters
    ----------
    lam : ndaray
        Wavelength range in microns (um).
        
    theta : float
        Angle of incidence in radians (rad).
        
    tfilm : float
        Film Thickness in milimiters (mm).
        
    Nlayer : tuple
        Refractive index above, in, and below the film. Length of 
        N must be 3.
        
    fv : float
        Particle's volume fraction.
        
    D : float
        Particle's diameter in microns (um)
    
    Np : ndarray or float
        Refractive index of particles. If ndarray, the size must be equal to
        len(lam)

    Returns
    -------    
    Ttot : ndarray
        Spectral total transmisivity
        
    Rtot : ndarray
        Spectral total reflectivity
        
    Tspec : ndarray
        Spectral specular transmisivity

    '''
    if np.isscalar(lam): lam = np.array([lam]) # convert lam to ndarray

    assert isinstance(Nlayer, tuple), 'Nlayers must be on tuple format of dim = 3'
    assert len(Nlayer) == 3, 'length of Nlayer must be == 3'
    if not np.isscalar(Np):
        assert len(Np) == len(lam), 'Np must be either scalar or an ndarray of size len(lam)'
    
    # convert all refractive index to arrays of size len(lam)
    # store result into a list
    N = []
    for Ni in Nlayer:
        if np.isscalar(Ni): 
            Ni = np.ones(len(lam))*Ni
        else: 
            assert len(Ni) == len(lam), 'refractive index must be either scalar or ndarray of size len(lam)'
        N.append(Ni)
    
    tfilm = tfilm*1E3 # convert mm to micron units
    
    Rp, Tp = wv.incoh_multilayer(lam, theta, N, tfilm, pol = 'TM')
    Rs, Ts = wv.incoh_multilayer(lam, theta, N, tfilm, pol = 'TE')
    T    = (Ts + Tp)/2
    Rtot = (Rp + Rs)/2
    
    # Get extinction and scattering efficiency of the sphere
    qext, qsca = mie.scatter_efficiency(lam, N[1], Np, D)[:2]
    qabs = qext - qsca # absorption efficiency
    
    Ac = pi*D**2/4 # cross section area of sphere
    Vp = pi*D**3/6 # volume of sphere
    cabs = Ac*qabs # absorption cross section
    cext = Ac*qext # extinction cross section
    
    theta1 = wv.snell(N[0],N[1], theta)
    # theta1 = np.zeros(len(lam),dtype=complex)
    # for i in range(len(lam)):
    #     theta1[i] = wv.snell(N[0][i],N[1][i], theta)
        
    Ttot = T*exp(-fv/Vp*cabs*tfilm/cos(theta1.real))
    Tspec = T*exp(-fv/Vp*cext*tfilm/cos(theta1.real))
    
    return Ttot, Rtot, Tspec

def adm_sphere(lam,tfilm,Nlayers,fv,D,Np, diffuse = False):
    '''
    Reflectivitiy and transmissivity for a film with spherical particles. This 
    function considers multiple scattering using adding-doubling method (adm) from 
    iadpython library.

    Parameters
    ----------
    lam : ndaray
        Wavelength range in microns.
        
    tfilm : float
        Film Thickness in milimiters.
        
    Nlayers : tuple
        Refractive index above, in, and below the film. Length of 
        N must be 3.
        
    fv : float
        Particle's volume fraction.
        
    D : float
        Particle's diameter in microns
    
    Np : ndarray or float
        Refractive index of particles. If ndarray, the size must be equal to
        len(lam)

    diffuse: bool
        Set diffuse approximation. If True, gcos = 0

    Returns
    -------    
    Ttot : ndarray
        Spectral total transmisivity

    Rtot : ndarray
        Spectral total reflectivity    

    Tspec : ndarray
        Spectral specular transmisivity
        
    Rspec : ndarray
        Spectral specular reflectivity
    '''
    # here we only verify that length of Nlayers is 3
    assert isinstance(Nlayers, tuple), 'Nlayers must be on tuple format of dim = 3'
    assert len(Nlayers) == 3,          'Number of layers must be 3'

    # Verify D is float or list
    #   1. solid sphere
    if np.isscalar(D) : D = [D,]
    #   2. multilayered sphere
    else:
        assert isinstance(D, list), 'diameter of shell layers must be on a list format'

    # get particle's concentration
    Vp = np.pi*D[-1]**3/6  # particle volume (um^3)
    Ac = np.pi*D[-1]**2/4  # cross section area (um2)

    # unpack refractive index of layers
    N_up, Nh, N_dw = Nlayers

    # get scattering and extinction efficiencies and asymmetry parameter
    qext, qsca, gcos = mie.scatter_efficiency(lam,Nh,Np,D)

    # get absorption efficiency
    qabs = qext - qsca
    qabs[qabs < 0] = 0 # if absorption is negative, make it zero
    
    # convert results to cross sections
    Csca = qsca*Ac     # scattering cross section (um2)
    Cext = qext*Ac     # extinction cross section (um2)
    Cabs = qabs*Ac     # absorption cross section (um2)

    # compute scattering and absorption coefficients
    k_sca = fv/Vp*Csca
    k_abs = fv/Vp*Cabs

    if diffuse: gcos = gcos*0

    return adm(lam,tfilm, k_sca, k_abs, gcos, Nh, N_up, N_dw)

@np.vectorize
def adm(lam,tfilm, k_sca, k_abs, gcos, Nh, Nup=1.0, Ndw=1.0):
    '''
    Reflectivitiy and transmissivity for a film with particles of arbitrary shape. This 
    function considers multiple scattering using adding-doubling method (adm) from 
    iadpython library.

    Parameters
    ----------
    lam : ndaray
        Wavelength range in microns.
        
    tfilm : float
        Film Thickness in milimiters.
        
    k_sca : ndarray
        Scattering coefficient  = fv/Vp*Csca (um^-1)

    k_abs : ndarray
        Absorption coefficient = fv/Vp*Cabs (um^-1)
        
    gcos : ndarray
        Assymmetry parameter

    Nh : float
        Refractive index of host medium

    N_up : float, optional
        Refractive index above film. Default is 1.0
        
    N_dw : float, optional
        Refractive index below film. Default is 1.0

    Returns
    -------    
    Ttot : ndarray
        Spectral total transmisivity

    Rtot : ndarray
        Spectral total reflectivity    

    Tspec : ndarray
        Spectral specular transmisivity
        
    Rspec : ndarray
        Spectral specular reflectivity

    '''
    kz_imag = 2*np.pi/lam*Nh.imag*1E3   # imaginary part of wavevector (mm^-1)
    mu_s = k_sca*1E3                    # scattering coefficient (mm^-1) 
    mu_a = k_abs*1E3 + 2*kz_imag        # absorption coefficient (mm^-1)
    g = gcos                            # asymmetry parameter
    d = tfilm                           # film thickness (mm)
    
    if mu_s == 0 and mu_a == 0: a, b = 0, 0
    else:
        a = mu_s/(mu_a+mu_s)
        b = (mu_a+mu_s)*d
        
    if Nh.real < 1: Nh = 1             # iad throws error if Nh < 1 (Need fix iad: PENDING)

    # Set sample for adding-doubling simulation
    s = iad.Sample(a=a, b=b, g=g, d=d,  
                   n=Nh.real, n_above=Nup.real, n_below=Ndw.real)
    
    # compute total components (only at normal incidence)
    R_tot, T_tot, R_tot_all, R_tot_all = s.rt() # discard components at all incidenct angles (_all)
    
    # compute specular (unscattered) componentes
    R_spec, T_spec = s.unscattered_rt()

    return T_tot, R_tot, T_spec, R_spec    
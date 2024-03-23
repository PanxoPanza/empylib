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
    
    theta1 = np.zeros(len(lam),dtype=complex)
    for i in range(len(lam)):
        theta1[i] = wv.snell(N[0][i],N[1][i], theta)
        
    Ttot = T*exp(-fv/Vp*cabs*tfilm/cos(theta1.real))
    Tspec = T*exp(-fv/Vp*cext*tfilm/cos(theta1.real))
    
    return Ttot, Rtot, Tspec

def adm_sphere(lam,tfilm,Nlayers,fv,D,Np):
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
    assert len(Nlayers) == 3,         'Number of layers must be 3'

    # get scattering efficiency and asymmetry parameter
    Nh = Nlayers[1]
    qext, qsca, gcos = mie.scatter_efficiency(lam,Nh,Np,D)
    
    # convert results to cross sections
    Ac = np.pi*D**2/4  # cross section area (um2)
    Csca = qsca*Ac     # scattering cross section (um2)
    Cext = qext*Ac     # extinction cross section (um2)
    Cabs = Cext - Csca # absorption cross section (um2)
    Vp = np.pi*D**3/6  # particle volume (um^3)

    return adm(lam,tfilm,Nlayers,fv,Csca,Cabs,gcos,Vp)

def adm(lam,tfilm,Nindex,fv,Csca,Cabs,gcos,Vp):
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
        
    Nindex : tuple
        Refractive index above, in, and below the film. Length of N must be 3.
        
    fv : float
        Particle's volume fraction.
        
    Csca : ndarray
        Scattering cross section (um^2)

    Cabs : ndarray
        Absorption cross section (um^2)
        
    gcos : ndarray
        Assymmetry parameter
    
    Vp : float
        Effective volume of particle (um^3)

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
    if np.isscalar(lam): lam = np.array([lam]) # convert lam to ndarray
    
    # analize refractive index of layers
    assert isinstance(Nindex, tuple), 'Nindex must be a tuple of dim = 3'
    assert len(Nindex) == 3, 'length of Nindex must be == 3'

    N = []
    for Ni in Nindex:
        if np.isscalar(Ni): 
            Ni = np.ones(len(lam))*Ni
        else: 
            assert len(Ni) == len(lam), 'Nindex must either float or size len(lam)'
        N.append(Ni)
    
    Nabove, Nh, Nbelow = N
    
    # check scattering and absorption cross sections of particle
    # ........... Scattering
    if np.isscalar(Csca): 
       Csca = np.ones(len(lam))*Csca
    else: 
        assert len(Csca) == len(lam), 'Csca must either float or size len(lam)'
        
    # ........... Absorption
    if np.isscalar(Cabs): 
       Cabs = np.ones(len(lam))*Cabs
    else: 
        assert len(Cabs) == len(lam), 'Cabs must either float or size len(lam)'
        
    # ........... Asymmetry parameter
    if np.isscalar(gcos): 
       gcos = np.ones(len(lam))*gcos
    else: 
        assert len(gcos) == len(lam), 'gcos must either float or size len(lam)'

    # iterate using iadpython
    Ttot = np.empty_like(lam)
    Rtot = np.empty_like(lam)
    Tspec = np.empty_like(lam)
    Rspec = np.empty_like(lam)
    for i in range(len(lam)):
        kz_imag = 2*np.pi/lam[i]*Nh[i].imag*1E3 # imaginary part of wavevector (mm^-1)
        mu_s = fv*Csca[i]/Vp*1E3                # scattering coefficient (mm^-1) 
        mu_a = fv*Cabs[i]/Vp + 2*kz_imag        # absorption coefficient (mm^-1)
        g = gcos[i]                             # asymmetry parameter
        d = tfilm                               # film thickness (mm)
        
        if mu_s == 0 and mu_a == 0: a, b = 0, 0
        else:
            a = mu_s/(mu_a+mu_s)
            b = (mu_a+mu_s)*d
        
        # Set sample for adding-doubling simulation
        s = iad.Sample(a=a, b=b, g=g, d=d,  
                       n=Nh[i].real, n_above=Nabove[i].real, n_below=Nbelow[i].real)

        # compute total components (only at normal incidence)
        r_tot_1, t_tot_1, r_tot_all, t_tot_all = s.rt() # discard components at all incidenct angles 

        # compute specular (unscattered) componentes
        r_spec, t_spec = s.unscattered_rt()
        
        Rtot[i] = r_tot_1
        Ttot[i] = t_tot_1
        Rspec[i] = r_spec
        Tspec[i] = t_spec
        
    return Ttot, Rtot, Tspec, Rspec    
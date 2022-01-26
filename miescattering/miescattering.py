# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 23:38:11 2021

@author: frami
"""
import numpy as np
from numpy import pi
from scipy.special import jv, yv

def miecoated(m,x, nmax = -1):
    
    # size parameter for the outer diameter of the sphere
    ka = x[-1]
    
    if nmax == -1 :
        # define nmax according to B.R Johnson (1996)
        nmax = np.round(abs(ka) + 4*abs(ka)**(1/3) + 2)
    
    mix = m*x               # Ni*k*ri
    mi1 = np.append(m,1)
    mi1x = mi1[1:]*x        # Ni+1*k*ri
    
    # Computation of Dn(z), Gn(z) and Rn(z)
    nmx = np.round(max(nmax, max(abs(m*x))) + 16)
    
    # Get Dn(mi*x), Gn(mi*x), Rn(mi*x) 
    Dn, Gn, Rn = log_RicattiBessel(mix,nmax,nmx)
    
    # Get Dn(mi+1*x), Gn(mi+1*x), Rn(mi+1*x)
    Dn1, Gn1, Rn1 = log_RicattiBessel(mi1x,nmax,nmx)
    
    # Get an and bn
    an, bn = recursive_ab(mi1, Dn, Gn, Rn, 
                          Dn1, Gn1, Rn1, nmax)
    
    # 
    

def miescat(lam,Nh,Np,D):
    
    '''
    Compute scattering properties for a single sphere

    Parameters
    ----------
    lam : 1D numpy array
        wavelengtgh (um)
    Nh : 1D numpy array
        Complex refractive index of host
    Np : numpy array (t x lam)
        Complex refractive index of sphere
    D : 1D array (size t)
        Diameter of sphere shells

    Returns
    -------
    Qabs : Absorption efficiency
    Qsca : Scattering efficiency
    gcos : Asymmetry parameter
    '''
    assert len(lam) == Np.shape[1]
    assert len(D) == Np.shape[0]
    
    m = Np/Nh           # sphere layers
    R = D/2             # particle's inner radius
    kh = 2*pi*Nh/lam    # wavector in the host
    x = kh*R            # size parameter
    
    m = m.transpose()
    x = x.transpose()
    for iw in range(len(lam)) :
        Qext, Qsca, gcos = miecoated(m[iw,:],x[iw,:])
    
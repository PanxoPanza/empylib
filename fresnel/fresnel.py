# -*- coding: utf-8 -*-
"""
Library of reflection a transnmission functions

Created on Sun Nov  7 17:25:53 2021

@author: PanxoPanza
"""
import numpy as np
from numpy import meshgrid, cos, sin, sqrt, conj, real, abs, pi

def interface(theta,n1,n2):
    '''
    This function computes the Fresnel coeficients at an interface
    
    Parameters
    ----------
    theta : np.array 1-D
        Angle of incidence in degrees
    n1 : np.array 1-D
        spectral refractive index medium 1
    n2 : np.array 1-D
        spectral refractive index medium 2

    Returns
    -------
    r_p: reflection coeficient TM
    t_p: transmission coeficient TM
    R_p: reflectivity TM
    T_p: transmissivity TM
    r_s: reflection coeficient TE
    t_s: transmission coeficient TE
    R_s: reflectivity TE
    T_s: transmissivity TE
    '''
    theta = np.radians(theta) # conver degrees to radians
    
    nn1, tt = meshgrid(n1,theta)
    nn2     = meshgrid(n2,theta)[0]
    
    sin_Ti = sin(tt)
    cos_Ti = cos(tt)
    sin_Tt = nn1*sin_Ti/nn2
    cos_Tt = sqrt(1 - sin_Tt**2)
    
    # compute p polarization (TM)
    r_p = (nn1*cos_Tt - nn2*cos_Ti)/            \
          (nn1*cos_Tt + nn2*cos_Ti)
          
    t_p = (2*nn1*cos_Ti)/                       \
          (nn1*cos_Tt + nn2*cos_Ti)
    
    R_p = r_p*conj(r_p)
    T_p = real(conj(nn2)*cos_Tt)/                 \
          real(conj(nn1)*cos_Ti)                  \
          *t_p*conj(t_p)
    
    R_p = abs(R_p)
    T_p = abs(T_p)
    
    # compute s polarization (TE)
    r_s = (nn1*cos_Ti - nn2*cos_Tt)/             \
          (nn1*cos_Ti + nn2*cos_Tt)
          
    t_s = (2*nn1*cos_Ti)/                        \
          (nn1*cos_Ti + nn2*cos_Tt)
    
    R_s = r_s*conj(r_s)
    T_s = real(nn2*cos_Tt)/                 \
          real(nn1*cos_Ti)                  \
          *t_p*conj(t_s)
        
    R_s = abs(R_s)
    T_s = abs(T_s)
    
    # if theta is not an array reshape array to a column vector (n,)
    if not type(theta) == np.ndarray :
        return R_p.reshape(-1,), T_p.reshape(-1,), \
               r_p.reshape(-1,), t_p.reshape(-1,), \
               R_s.reshape(-1,), T_s.reshape(-1,), \
               r_s.reshape(-1,), t_s.reshape(-1,), \
        
        
    return R_p, T_p, r_p, t_p, \
           R_s, T_s, r_s, t_s

def multilayer(lam,tt,N,d, pol):
    '''
    Get Fresnel coeficients and energy flux of multilayered films

    Parameters
    ----------
    lam : TYPE
        DESCRIPTION.
    tt : TYPE
        DESCRIPTION.
    N0 : TYPE
        DESCRIPTION.
    Nend : TYPE
        DESCRIPTION.
    N : TYPE
        DESCRIPTION.
    d : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    # optical constant
    Z0 = 376.730
    tt = np.radians(tt) # transform angle from deg --> rad

    kk = 2*pi/lam
    # prepare variables (consider array of angles and frequencies)
    k0, theta = meshgrid(kk,tt)
    N0   = meshgrid(N[0],tt)[0]
    Nend = meshgrid(N[-1],tt)[0]

    # optical properties of the semi-infinite media
    # semi-infinite media left 
    sinT0 = sin(theta)
    cosT0 = sqrt(1 - sinT0**2)+1E-15

    # semi-infinite media right
    sinTn = N0*sinT0/Nend
    cosTn = sqrt(1 - sinTn**2)+1E-15

    if 's' in pol :
        P0 = - Z0/(N0*cosT0)
        Pn = - Z0/(Nend*cosTn)
        Ct = 1
    elif 'p' in pol : 
        P0 =   Z0*cosT0/N0
        Pn =   Z0*cosTn/Nend
        Ct = cosT0/cosTn

    # Calculate transfer matrix
    m11 = 1 
    m12 = 0 
    m21 = 0 
    m22 = 1
    
    nLayers = len(N)-2
    for i in range(nLayers):
        Ni = meshgrid(N[i+1],tt)[0]
        sinTi = N0*sinT0/Ni
        cosTi = sqrt(1 - sinTi**2)+1E-10
        kzd = Ni*k0*cosTi*d[i]
        
        if 's' in pol :
            Pi = - Z0/(Ni*cosTi)
        elif 'p' in pol :
            Pi =   Z0*cosTi/Ni
        
        m11_new = +         cos(kzd)*m11 - 1j/Pi*sin(kzd)*m12
        m12_new = - 1j*Pi*sin(kzd)*m11 +         cos(kzd)*m12
        m21_new = +         cos(kzd)*m21 - 1j/Pi*sin(kzd)*m22
        m22_new = - 1j*Pi*sin(kzd)*m21 +         cos(kzd)*m22
        
        # store new values
        m11 = m11_new
        m12 = m12_new
        m21 = m21_new
        m22 = m22_new 
    
    # get reflection and tranmission coefficients
    r = ((Pn*m11 + m12) - (Pn*m21 + m22)*P0)/ \
        ((Pn*m11 + m12) + (Pn*m21 + m22)*P0)

    t = (2*Ct*Pn)/                             \
        ((Pn*m11 + m12) + (Pn*m21 + m22)*P0)

    # get reflectivity and transmissivity
    if 's' in pol:
        R = abs(r)**2
        T = real(Nend *cosTn)/real(N0 *cosT0)*abs(t)**2
    elif 'p' in pol:
            R = abs(r)**2
            T = real(conj(Nend)*cosTn)/real(conj(N0)*cosT0)*abs(t)**2
    
    # if theta is not an array reshape array to a column vector
    if not type(tt) == np.ndarray :
        return R.reshape(-1,), T.reshape(-1,), \
               r.reshape(-1,), t.reshape(-1,)
        
    return R, T, r, t
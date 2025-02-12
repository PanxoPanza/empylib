# -*- coding: utf-8 -*-
"""
Library of wave optics funcions

Created on Sun Nov  7 17:25:53 2021

@author: PanxoPanza
"""

import numpy as np

def interface(theta,N1,N2, pol='TM'):
    '''
    Computes the Fresnel coeficients  and energy flux of at an interface. For
    each theta, this function will compute the Fresnel coeficients at the 
    spectrum defined for N1 or N2.
    
    Parameters
    ----------
    theta : ndarray or float
        Angle of incidence in radians.
        
    N1 : ndarray or float
        Spectral refractive index medium above the interface.
        
    N2 : ndarray or float
        Spectral refractive index medium below the interface.
        
    pol: str (optional)
        Polarization of incident field. Could be:
            - 'TM' transverse magnetic (default)
            - 'TE' transverse electric

    Returns
    -------
    R: ndarray
        Reflectivity
        
    T: ndarray
        Transmissivity
        
    r : ndarray
        Reflection coeficient
        
    t: ndarray
        Transmission coeficient
    
    '''
    from numpy import meshgrid, cos, sin, sqrt, conj, real, abs, pi, exp
    from numpy.lib.scimath import arcsin
    
    nn1, tt = meshgrid(N1,theta)
    nn2     = meshgrid(N2,theta)[0]
    
    sin_Ti = sin(tt)
    cos_Ti = cos(tt)
    sin_Tt = nn1*sin_Ti/nn2
    cos_Tt = sqrt(1 - sin_Tt**2)
    
    # compute p polarization (TM)
    if pol == 'TM' :
        r = (nn1*cos_Tt - nn2*cos_Ti)/            \
            (nn1*cos_Tt + nn2*cos_Ti)
          
        t = (2*nn1*cos_Ti)/                       \
            (nn1*cos_Tt + nn2*cos_Ti)
    
        R = r*conj(r)
        T = real(conj(nn2)*cos_Tt)/                 \
          real(conj(nn1)*cos_Ti)                  \
          *t*conj(t)
    
        R = abs(R)
        T = abs(T)
    
    # compute s polarization (TE)
    elif pol == 'TE':
        r = (nn1*cos_Ti - nn2*cos_Tt)/             \
            (nn1*cos_Ti + nn2*cos_Tt)
          
        t = (2*nn1*cos_Ti)/                        \
                (nn1*cos_Ti + nn2*cos_Tt)
    
        R = r*conj(r)
        T = real(nn2*cos_Tt)/                 \
              real(nn1*cos_Ti)                  \
                *t*conj(t)
        
        R = abs(R)
        T = abs(T)
    
    # if theta is not an array reshape array to a column vector (n,)
    if not type(theta) == np.ndarray :
        return R.reshape(-1,), T.reshape(-1,), \
               r.reshape(-1,), t.reshape(-1,)
        
        
    return R, T, r, t

def multilayer(lam,tt,N,d=(), pol='TM'):
    '''
    Get Fresnel coeficients and energy flux of multilayered films. The function 
    computes the spectral Fresnel coefficients at each angle of incidence

    Parameters
    ----------
    lam : ndaray or float
        Wavelength range in microns.
        
    tt : ndarray or float
        Angle of incidence in radians. 
        
    N : tuple
        Refractive index of each layers, including the medium above and below 
        the film (i.e., minimum 2 arguments). The number of elements must be 
        equal to len(d) + 2. Each element should be, either, a float or a 
        ndarray of size (n,), where n = len(lam)
        
    d : tuple
        Thickness of each layer in microns
        
    pol: str (optional)
        Polarization of incident field:
            - 'TM' transverse magnetic (default)
            - 'TE' transverse electric

    Returns
    -------
    R: ndarray
        Reflectivity
        
    T: ndarray
        Transmissivity
        
    r : ndarray
        Reflection coeficient
        
    t: ndarray
        Transmission coeficient

    '''
    from numpy import meshgrid, cos, sin, sqrt, conj, real, abs, pi, exp
    from numpy.lib.scimath import arcsin
    
    # configure input data and assert size compatibility
    lam, d = assert_multilayer_input(lam,d,N)
    
    # optical constant
    Z0 = 376.730

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

    if pol=='TE' :
        P0 = - Z0/(N0*cosT0)
        Pn = - Z0/(Nend*cosTn)
        Ct = 1
    elif pol=='TM': 
        P0 =   Z0*cosT0/N0
        Pn =   Z0*cosTn/Nend
        Ct = cosT0/cosTn

    # Calculate transfer matrix
    m11 = 1 
    m12 = 0 
    m21 = 0 
    m22 = 1
    
    nLayers = len(d)
    for i in range(nLayers):
        Ni = meshgrid(N[i+1],tt)[0]
        sinTi = N0*sinT0/Ni
        cosTi = sqrt(1 - sinTi**2)+1E-10
        kzd = Ni*k0*cosTi*d[i]
        
        if   pol=='TE' :
            Pi = - Z0/(Ni*cosTi)
        elif pol=='TM' :
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
    if pol=='TE' :
        R = abs(r)**2
        T = real(Nend *cosTn)/real(N0 *cosT0)*abs(t)**2
    elif pol=='TM' :
        R = abs(r)**2
        T = real(conj(Nend)*cosTn)/real(conj(N0)*cosT0)*abs(t)**2
    
    # if theta is not an array reshape array to a column vector
    if not type(tt) == np.ndarray :
        return R.reshape(-1,), T.reshape(-1,), \
               r.reshape(-1,), t.reshape(-1,)
        
    return R, T, r, t

def incoh_multilayer(lam,theta,Nlayer,d=(),pol='TM', coh_length=0):
    '''
    Transfer matrix method (TMM) for coherent, and incoherent multilayer 
    structures. (Only tested at normal incidence)
    
    Source: Katsidis, C. C. & Siapkas, D. I. Appl. Opt. 41, 3978 (2002).

    Parameters
    ----------
    lam : ndarray
        wavelength range (microns)
        
    theta : float
        angle of incidence (radian).
        
    Nlayer : tuple
        Refractive index of each layers, including the medium above and below 
        the film (i.e., minimum 2 arguments). The number of elements must be 
        equal to len(d) + 2. Each element should be, either, a float or a 
        ndarray of size (n,), where n = len(lam)
        
    d : tuple
        Thickness of each layer in microns
        
    pol: str (optional)
        Polarization of incident field:
            - 'TM' transverse magnetic (default)
            - 'TE' transverse electric
            
    coh_length: float
        Coherence length of the source (microns), 0 by default

    Returns
    -------
    R: ndarray
        Reflectivity
        
    T: ndarray
        Transmissivity

    '''
    from numpy import meshgrid, cos, sin, sqrt, conj, real, abs, pi, exp
    from numpy.lib.scimath import arcsin

    # configure input data and assert size compatibility
    lam, d = assert_multilayer_input(lam,d,Nlayer)
    
    Lc = coh_length

    # check layers with thickenss larger than Lc
    is_incoherent = d > Lc*cos(theta)/2;
    
    # convert all refractive index to arrays of size len(lam)
    # store result into a list
    N = []
    for Ni in Nlayer:
        if np.isscalar(Ni): 
            Ni = np.ones(len(lam))*Ni
        else: 
            assert len(Ni) == len(lam), 'refractive index must be either scalar or ndarray of size len(lam)'
        N.append(Ni)
    
    Nmid = [N[0]]
    dmid = []

    TintP11 = 1
    TintP12 = 0
    TintP21 = 0
    TintP22 = 1
    
    nLayers = len(d)
    th_0 = theta*np.ones(len(lam))
    for m in range(nLayers):
    
        # Compute transfer matrix for incoherent layers
        #----------------------------------------------------------------------
        if is_incoherent[m] :
            Nmid.append(N[m+1])
            
            T11_coh, T12_coh, T21_coh, T22_coh, th_end = \
                TMMcoh(lam, th_0, Nmid, dmid, pol)
            
            Tint_11 = TintP11*T11_coh + TintP12*T21_coh
            Tint_12 = TintP11*T12_coh + TintP12*T22_coh
            Tint_21 = TintP21*T11_coh + TintP22*T21_coh
            Tint_22 = TintP21*T12_coh + TintP22*T22_coh
            
            kzd = 2*pi/lam*N[m+1]*cos(th_end)*d[m]
            exp_2kd = exp(-2*kzd.imag)
            
            # restrict small values to avoid overflow
            exp_2kd = exp_2kd*(exp_2kd>= 1e-30) + (exp_2kd< 1e-30)*1e-30 
            
            P11 = 1/exp_2kd
            P12 = 0
            P21 = 0
            P22 = exp_2kd
            
            TintP11 = Tint_11*P11 + Tint_12*P21
            TintP12 = Tint_11*P12 + Tint_12*P22
            TintP21 = Tint_21*P11 + Tint_22*P21
            TintP22 = Tint_21*P12 + Tint_22*P22
        
            th_0 = th_end   # update angle of incidence
            Nmid = [N[m+1]]
            dmid = []
        else :
            Nmid.append(N[m+1])
            dmid.append(d[m])

    
    Nmid.append(N[-1])
    
    T11_coh, T12_coh, T21_coh, T22_coh, th_end = \
        TMMcoh(lam, th_0, Nmid, dmid, pol)

    Tint_11 = TintP11*T11_coh + TintP12*T21_coh
    Tint_12 = TintP11*T12_coh + TintP12*T22_coh
    Tint_21 = TintP21*T11_coh + TintP22*T21_coh
    Tint_22 = TintP21*T12_coh + TintP22*T22_coh

    R = Tint_21/Tint_11
    T = 1/Tint_11
    
    # get reflectivity and transmissivity
    if pol=='TE' :
        R = R
        T = real(     N[-1] *cos(th_end))/  \
            real(     N[ 0] *cos(theta))*T
    elif pol=='TM' :
        R = R
        T = real(conj(N[-1])*cos(th_end))/  \
            real(conj(N[ 0])*cos(theta))*T

    
    return R, T

def assert_multilayer_input(lam,d,N):
    '''
    Verify that multilayer input complies with required dimensions

    Parameters
    ----------
    lam : float or ndarray
        wavelength range (microns)
        
    d : float or tuple
        Thickness of each layer in microns.
        
    N : tuple
        Refractive index of each layers, including the medium above and below 
        the film (i.e., minimum 2 arguments). The number of elements must be 
        equal to len(d) + 2. Each element should be, either, a float or a 
        ndarray of size (n,), where n = len(lam)

    Returns
    -------
    lam : ndarray
        wavelength range (microns)
        
    d : tuple
        Thickness of each layer in microns

    '''
    from numpy import meshgrid, cos, sin, sqrt, conj, real, abs, pi, exp
    from numpy.lib.scimath import arcsin
        
    if np.isscalar(lam) :  lam = np.array([lam]) # convert lam to a ndarray
    if np.isscalar(d):  d = (d,)              # convert d to an iterable tuple
    
    assert len(N) == len(d) + 2, 'number of elements in N must be len(d) + 2'
    for Nlayer in N:
        if not np.isscalar(Nlayer):
            assert len(Nlayer) == len(lam), \
                'each refractive must be either a float or an ndarray of size len(lam)'
    
    return lam, d

def TMMcoh(lam, th_0, Nmid, dmid, pol):
    import numpy as np
    from numpy import meshgrid, cos, sin, sqrt, conj, real, abs, pi, exp
    from numpy.lib.scimath import arcsin

    # store layer thickness
    dfw = dmid.copy()
    dbw = dmid.copy()
    dbw.reverse()
    
    # adjust angle of incidence to the incoherent layer
    # and iterate over each wavelength
    th_end = np.zeros(len(lam),dtype=complex)
    r0m = np.zeros(len(lam),dtype=complex)
    t0m = np.zeros(len(lam),dtype=complex)
    rm0 = np.zeros(len(lam),dtype=complex)
    tm0 = np.zeros(len(lam),dtype=complex)
    
    for i in range(len(lam)):
        th_end[i] = snell(Nmid[0][i], Nmid[-1][i], th_0[i])
        
        Nfw = []
        Nbw = []
        for j in range(len(Nmid)):
            Nfw.append(Nmid[     j][i])
            Nbw.append(Nmid[-(j+1)][i])
        
        r0m[i],t0m[i] = multilayer(lam[i], th_0[i],Nfw, dfw,pol)[-2:]
        rm0[i],tm0[i] = multilayer(lam[i], th_end[i],Nbw, dbw,pol)[-2:]

    T11_coh = 1/abs(t0m)**2;         
    T12_coh = - abs(rm0)**2/abs(t0m)**2;
    T21_coh = + abs(r0m)**2/abs(t0m)**2; 
    T22_coh = (abs(t0m*tm0)**2 - abs(r0m*rm0)**2)/abs(t0m)**2;
    return T11_coh, T12_coh, T21_coh, T22_coh, th_end


#------------------------------------------------------------------
# These are extra functions from TMM python code by Steve Byrnes
# source https://github.com/sbyrnes321/tmm
#------------------------------------------------------------------
def is_forward_angle(n, theta):
    """
    if a wave is traveling at angle theta from normal in a medium with index n,
    calculate whether or not this is the forward-traveling wave (i.e., the one
    going from front to back of the stack, like the incoming or outgoing waves,
    but unlike the reflected wave). For real n & theta, the criterion is simply
    -pi/2 < theta < pi/2, but for complex n & theta, it's more complicated.
    See https://arxiv.org/abs/1603.02720 appendix D. If theta is the forward
    angle, then (pi-theta) is the backward angle and vice-versa.
    """
    from numpy import meshgrid, cos, sin, sqrt, conj, real, abs, pi, exp
    from numpy.lib.scimath import arcsin
    import sys
    EPSILON = sys.float_info.epsilon # typical floating-point calculation error


    assert n.real * n.imag >= 0, ("For materials with gain, it's ambiguous which "
                                  "beam is incoming vs outgoing. See "
                                  "https://arxiv.org/abs/1603.02720 Appendix C.\n"
                                  "n: " + str(n) + "   angle: " + str(theta))
    ncostheta = n * cos(theta)
    if abs(ncostheta.imag) > 100 * EPSILON:
        # Either evanescent decay or lossy medium. Either way, the one that
        # decays is the forward-moving wave
        answer = (ncostheta.imag > 0)
    else:
        # Forward is the one with positive Poynting vector
        # Poynting vector is Re[n cos(theta)] for s-polarization or
        # Re[n cos(theta*)] for p-polarization, but it turns out they're consistent
        # so I'll just assume s then check both below
        answer = (ncostheta.real > 0)
    # convert from numpy boolean to the normal Python boolean
    answer = bool(answer)
    # double-check the answer ... can't be too careful!
    error_string = ("It's not clear which beam is incoming vs outgoing. Weird"
                    " index maybe?\n"
                    "n: " + str(n) + "   angle: " + str(theta))
    if answer is True:
        assert ncostheta.imag > -100 * EPSILON, error_string
        assert ncostheta.real > -100 * EPSILON, error_string
        assert (n * cos(theta.conjugate())).real > -100 * EPSILON, error_string
    else:
        assert ncostheta.imag < 100 * EPSILON, error_string
        assert ncostheta.real < 100 * EPSILON, error_string
        assert (n * cos(theta.conjugate())).real < 100 * EPSILON, error_string
    return answer

@np.vectorize
def snell(n_1, n_2, th_1):
    """
    return angle theta in layer 2 with refractive index n_2, assuming
    it has angle th_1 in layer with refractive index n_1. Use Snell's law. Note
    that "angles" may be complex!!
    """
    from numpy.lib.scimath import arcsin
    # Important that the arcsin here is numpy.lib.scimath.arcsin, not
    # numpy.arcsin! (They give different results e.g. for arcsin(2).)
    th_2_guess = arcsin(n_1*np.sin(th_1) / n_2)
    if is_forward_angle(n_2, th_2_guess):
        return th_2_guess
    else:
        return pi - th_2_guess

def list_snell(n_list, th_0):
    """
    return list of angle theta in each layer based on angle th_0 in layer 0,
    using Snell's law. n_list is index of refraction of each layer. Note that
    "angles" may be complex!!
    """
    from numpy.lib.scimath import arcsin
    # Important that the arcsin here is numpy.lib.scimath.arcsin, not
    # numpy.arcsin! (They give different results e.g. for arcsin(2).)
    print('n_list[0]', n_list[0])
    print('n_list',n_list)
    angles = arcsin(n_list[0]*np.sin(th_0) / n_list)
    # The first and last entry need to be the forward angle (the intermediate
    # layers don't matter, see https://arxiv.org/abs/1603.02720 Section 5)
    if not is_forward_angle(n_list[0], angles[0]):
        angles[0] = pi - angles[0]
    if not is_forward_angle(n_list[-1], angles[-1]):
        angles[-1] = pi - angles[-1]
    return angles
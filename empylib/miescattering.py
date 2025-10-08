# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 23:38:11 2021

@author: PanxoPanza
"""
import numpy as _np
from numpy import pi, exp, conj, imag, real, sqrt
from scipy.special import jv, yv
from .nklib import emt_brugg
from .utils import _check_mie_inputs
import pandas as pd
from typing import Union as _Union, Optional as _Optional

def _log_RicattiBessel(x,nmax,nmx):
    '''
    Computes the logarithmic derivatives of Ricatti-Bessel functions,
        Dn(x) = psi_n'(x) / psi_n(x),
        Gn(x) = chi_n'(x) / chi_n(x), and
        Rn(x) = psi_n(x)  / xi_n(x);
    using the method by Wu & Wang Radio Sci. 26, 1393–1401 (1991).

    Parameters
    ----------
    x : 1D numpy array
        size parameter for each shell
    nmax : int
        number of mie coefficients
    nmx : int
        extended value of nmax for downward recursion (Wu & Wang, 1991)

    Returns
    -------
    1D numpy array
        Dn(x)
    1D numpy array
        Gn(x)
    1D numpy array
        Rn(x)
    '''
    
    # if x is scalar, transform variable to numpy array of dim 1
    if _np.isscalar(x): x = _np.array([x])
    
    n = _np.array(range(nmax))
    
    # Get Dn(x) by downwards recurrence
    Dnx = _np.zeros((len(x),nmx),dtype=_np.complex128)
    for i in reversed(range(1, nmx)):
        # define D_(i+1) (x)
        # if i == nmx-1 : Dip1 = _np.zeros(len(x))
        # else :          Dip1 = Dnx[:,i+1]
        
        Dnx[:,i-1] = (i+1)/x - 1/(Dnx[:,i] + (i+1)/x)
        
    # Get Gn(x) by upwards recurrence
    Gnx = _np.zeros((len(x),nmx),dtype=_np.complex128)
    G0x = 1j*_np.ones_like(x)
    i = 0
    Gnx[:,i] = 1/((i+1)/x - G0x) - (i+1)/x
    for i in range(1, nmx):
        # define G_(i-1) (x)
        # if i == 0 : Gim1x = 1j*_np.ones(len(x))
        # else : Gim1x = Gnx[:,i-1] 
        
        Gnx[:,i] = 1/((i+1)/x - Gnx[:,i-1]) - (i+1)/x
    
    # Get Rn(x) by upwards recurrence
    Rnx = _np.zeros((len(x),len(n)),dtype=_np.complex128) 
    for ix in range(len(x)):
        
        # note that 0.5*(1 - exp(-2j*x)) = 0 if x = pi*n
        # I added this clause for those cases
        if imag(x[ix]) == 0 and _np.mod(real(x[ix]),pi) == 0:
            nu = (n + 1) + 0.5
            py =  sqrt(0.5*pi*x[ix])*jv(nu,x[ix])
            chy = sqrt(0.5*pi*x[ix])*yv(nu,x[ix])
            gsy = py + 1j*chy
            Rnx[ix,:] = py/gsy
        
        # otherwise just do normal upward recursion
        else :            
            for i in range(nmax):
                if i == 0 : Rim1x = 0.5*(1 - exp(-2j*x[ix]))
                else :      Rim1x = Rnx[ix,i-1]
                
                Rnx[ix,i] = Rim1x*(Gnx[ix,i] + (i + 1)/x[ix])/  \
                                  (Dnx[ix,i] + (i + 1)/x[ix]) 
                                  
    return Dnx[:,n], Gnx[:,n], Rnx[:,n]

def _recursive_ab(m, n, Dn, Gn, Rn, Dn1, Gn1, Rn1) :
    i = Dn.shape[0]
    if i == 0:
        an = _np.zeros(n)
        bn = _np.zeros(n)
    else:
        # get an^i and bn^i
        (an, bn) = _recursive_ab(m[:i],n,
                                Dn[:i-1,:], Gn[:i-1,:], Rn[:i-1,:],
                                Dn1[:i-1,:],Gn1[:i-1,:],Rn1[:i-1,:])
        
        # get Un(mi*kri), Vn(mi, kri)
        Un = (Rn[i-1,:]*Dn[i-1,:] - an*Gn[i-1,:])/ \
                (Rn[i-1,:] - an + 1E-10)
        Vn = (Rn[i-1,:]*Dn[i-1,:] - bn*Gn[i-1,:])/ \
                (Rn[i-1,:] - bn + 1E-10)
        
        # get an^(i+1), bn^(i+1) by recursion formula
        an = Rn1[i-1,:]*(m[i]/m[i-1]*Un - Dn1[i-1,:])/ \
                        (m[i]/m[i-1]*Un - Gn1[i-1,:])
                      
        bn = Rn1[i-1,:]*(Vn - m[i]/m[i-1]*Dn1[i-1,:])/ \
                        (Vn - m[i]/m[i-1]*Gn1[i-1,:])

    return an, bn
        
def _get_coated_coefficients(m,x, nmax=None):
    '''
    Compute the mie coefficients an and bn using recursion algorithm from
    Johnson, Appl. Opt. 35, 3286 (1996).

    Parameters
    ----------
    m : 1D numpy array
        normalized refractive index of shell's layers
    x : 1D numpy array
        size parameter of shell's layers'
    nmax : int
        max number of expansion coefficients.

    Returns
    -------
    an : 1D numpy array (size nmax)
         mie coefficient for M function.
    bn : 1D numpy array (size nmax)
        mie coefficient for N function.
    phi : 1D numpy array (size nmax)
        1st order Bessel-Ricatti function (evaluated at ka).
    Dn1 : 1D numpy array (size nmax)
        Derivative of 1st order Bessel-Ricatti function (evaluated at ka).
    xi : 1D numpy array (size nmax)
        3rd order Bessel-Ricatti function (evaluated at ka).
    Gn1 : 1D numpy array (size nmax)
        Derivative of 2nd order Bessel-Ricatti function (evaluated at ka).

    '''
    assert len(x) == len(m)
    
    ka = x[-1] # size parameter of outer layer

    # define nmax according to B.R Johnson (1996)
    if nmax is None :
        nmax = int(_np.round(_np.abs(ka) + 4*_np.abs(ka)**(1/3) + 2))
    
    #----------------------------------------------------------------------
    #       Computing an and bn (main part of this code)
    #----------------------------------------------------------------------
    
    mix = m*x               # Ni*k*ri
    mi1 = _np.append(m,1)
    mi1x = mi1[1:]*x        # Ni+1*k*ri
    
    # Computation of Dn(z), Gn(z) and Rn(z)
    nmx = int(_np.round(max(nmax, max(abs(m*x))) + 16))
    
    # Get Dn(mi*x), Gn(mi*x), Rn(mi*x) 
    Dn, Gn, Rn = _log_RicattiBessel(mix,nmax,nmx)
    
    # Get Dn(mi+1*x), Gn(mi+1*x), Rn(mi+1*x)
    Dn1, Gn1, Rn1 = _log_RicattiBessel(mi1x,nmax,nmx)
    
    # Get an and bn
    an, bn = _recursive_ab(mi1, nmax, Dn, Gn, Rn, 
                                     Dn1, Gn1, Rn1)
    
    # ---------------------------------------------------------------------
    #       computing secondary paramters
    # ---------------------------------------------------------------------
    # Get Bessel-Ricatti functions and derivatives for last shell layer
    n = _np.array(range(1,nmax+1))
    nu = n+0.5
    phi = _np.sqrt(0.5*pi*ka)*jv(nu,ka) # phi(n,ka)
    chi = _np.sqrt(0.5*pi*ka)*yv(nu,ka) # chi(n,ka)
    xi  = phi + 1j*chi                    # xi(n,ka)
    
    return an.reshape(-1), bn.reshape(-1), phi, Dn1[-1,:].reshape(-1), xi, Gn1[-1,:].reshape(-1)

def _cross_section_at_lam(m,x,nmax = None):
    '''
    NEED TO CHECK FLUCTUATION FOR LARGE PARTICLES (F. RAMIREZ 2024)
    Compute mie scattering parameters for a given lambda
    The absorption, scattering, extinction and asymmetry parameter are 
    computed with the formulas for absorbing medium reported in 
    
    - Johnson, B. R. Light scattering by a multilayer sphere (1996). App. Opt., 
        35(18), 3286.
    
    - Wu, Z. S.; Wang, Y. P. (1991). Electromagnetic scattering for 
        multilayered sphere: Recursive algorithms. Science, 26(6), 1393–1401.

    Parameters
    ----------
    m : 1D numpy array
        normalized refractive index of shell layers
    x : 1D numpy array
        size paramter for each shell layer
    nmax : int, optional
        number of mie coefficients. The default is -1.

    Returns
    -------
    Qext : float
        Extinction efficiency.
    Qsca : float
        Scattering efficiency.
    Asym : float (-1, 1)
        Asymmetry parameter.
    Qb : float
        Backward scattering effiency.
    Qf : float
        Forward scatttering efficiency.
    nmax : int
        number of mie coefficients.
    '''
    assert len(x) == len(m)
    
    # determine nmax 
    y = x[-1] # size parameter of outer layer

    if nmax is None :
        # define nmax according to B.R Johnson (1996)
        nmax = int(_np.round(_np.abs(y) + 4*_np.abs(y)**(1/3) + 2))

    #------------------------------------------------------------------
    # Get mie coefficient and other parameters of interest
    #------------------------------------------------------------------
    (an, bn, py, Dy, xy, Gy) = _get_coated_coefficients(m,x,nmax)

    if imag(y) > 1E-8 :
        imy = 2*imag(y)
        ft = imy**2/(1 + (imy - 1)*exp(imy))
    else:
        ft = 2
    
    # arranging pre-computing constants
    n = _np.array(range(1,nmax+1))
    
    #------------------------------------------------------------------
    # Extinction efficiency
    #------------------------------------------------------------------
    en = (2*n+1)*imag((- 2j*py*conj(py)*imag(Dy)         \
                       + conj(an)*conj(xy)*py*Dy         \
                       - conj(bn)*conj(xy)*py*conj(Gy)   \
                       + an*xy*conj(py)*Gy               \
                       - bn*xy*conj(py)*conj(Dy))        \
                       /y)
    q = _np.sum(en)
    Qext = real(1/real(y)*ft*q)    
    
    #------------------------------------------------------------------
    # Scattering efficiency
    #------------------------------------------------------------------
    en = (2*n+1)*imag((+ _np.abs(an*xy)**2*Gy                \
                       - _np.abs(bn*xy)**2*conj(Gy)         \
                       )/y)
    q = _np.sum(en)
    Qsca = real(1/real(y)*ft*q)
    
    #------------------------------------------------------------------
    # Asymmetry parameter
    #------------------------------------------------------------------
    anp1 = _np.zeros(nmax,dtype=_np.complex128)
    bnp1 = _np.zeros(nmax,dtype=_np.complex128)
    anp1[:nmax-1] = an[1:] # a(n+1) coefficient
    bnp1[:nmax-1] = bn[1:] # a(n+1) coefficient
    
    asy1 = n*(n + 2)/(n + 1)*(an*conj(anp1)+ bn*conj(bnp1)) \
         + (2*n + 1)/(n*(n + 1))*real(an*conj(bn))
    
    asy2 = (2*n+1)*(an*conj(an) + bn*conj(bn))
    Asym = real(2*_np.sum(asy1)/_np.sum(asy2))
    
    #------------------------------------------------------------------
    # Backward scattering (not valid for absorbing host media)
    #------------------------------------------------------------------
    f = (2*n+1)*(-1)**n*(an - bn)
    q = _np.sum(f)
    Qb = real(q*conj(q)/y**2)
    
    #------------------------------------------------------------------
    # Forward scattering (not valid for absorbing host media)
    #------------------------------------------------------------------
    f = (2*n+1)*(an + bn)
    q = _np.sum(f)
    Qf = real(q*conj(q)/y**2)
    
    #------------------------------------------------------------------
    # Condition outputs to avoid unphysical results
    #------------------------------------------------------------------
    if Qsca < 0: Qsca = 0
    if Qext < Qsca: Qext = Qsca
    if Asym < -1: Asym = -1
    if Asym > +1: Asym = +1

    return Qext, Qsca, Asym, Qb, Qf

def scatter_efficiency(lam: _Union[float, _np.ndarray],
                       N_host: _Union[float, _np.ndarray],
                       Np_shells: _Union[float, _np.ndarray],
                       D: _Union[float, _np.ndarray],
                       nmax: int = None,
                       check_inputs: bool = True
                       ):

    '''
    Compute mie scattering parameters for multi-shell spherical particle.

    Parameters
    ----------
    lam : ndarray or float
        wavelengtgh (microns)
        
    N_host : ndarray or float
        Complex refractive index of host. If ndarray, its size must be equal to
        len(lam)
        
    Np_shells : float, 1darray or list
        Complex refractive index of each shell layer. The number of elements
        must be equal to len(D). Options are:
            float:   solid sphere and constant refractive index
            1darray: solid sphere and spectral refractive index (length must match that of lam)
            list:    multilayered sphere (with both constant or spectral refractive indexes)
        
    D : float or list
        Outter diameter of each shell's layer (microns). Options are:
            float: solid sphere
            list:  multilayered sphere

    nmax: int, optional  
        Number of mie scattering coefficients. Default None
    
    Returns
    -------
    Qext : ndarray
        Extinction efficiency
    Qsca : ndarray
        Scattering efficiency 
    gcos : ndarray
        Asymmetry parameter
    '''
    # first check inputs and arrange them in np arrays
    if check_inputs:
        lam, Nh, Np, D = _check_mie_inputs(lam,N_host,Np_shells,D)

    m = Np/Nh.real                  # sphere layers
    R = D/2                         # particle's inner radius
    kh = 2*pi*Nh.real/lam           # wavector in the host
    x = _np.tensordot(kh,R,axes=0)   # size parameter
    m = m.transpose()
    
    # Preallocate outputs
    Qext = _np.zeros_like(lam, dtype=float)
    Qsca = _np.zeros_like(lam, dtype=float)
    gcos = _np.zeros_like(lam, dtype=float)
    for i in range(len(lam)):
        Qext[i], Qsca[i], gcos[i], *_ = _cross_section_at_lam(m[i, :], x[i, :], nmax)
        
    # outputs: qext, qsca, gcos
    return Qext, Qsca, gcos
    
def scatter_coeffients(lam:_Union[float, _np.ndarray],
                       N_host:_Union[float, _np.ndarray],
                       Np_shells:_Union[float, _np.ndarray],
                       D:_Union[float, _np.ndarray],
                       nmax: int = None,
                       check_inputs: bool = True):
    
    '''
    Compute mie scattering coefficients an and bn for multi-shell spherical 
    object. Layeres must be sorted from inner to outter diameter

    Parameters
    ----------
    lam : ndarray or float
        wavelengtgh (microns)
        
    N_host : ndarray or float
        Complex refractive index of host. If ndarray, its size must be equal to
        len(lam)
        
    Np_shells : float, 1darray or list
        Complex refractive index of each shell layer. The number of elements
        must be equal to len(D). Options are:
            float:   solid sphere and constant refractive index
            1darray: solid sphere and spectral refractive index (length must match that of lam)
            list:    multilayered sphere (with both constant or spectral refractive indexes)
        
    D : float or list
        Outter diameter of each shell's layer (microns). Options are:
            float: solid sphere
            list:  multilayered sphere

    nmax: int, optional  
        Number of mie scattering coefficients. Default None

    Returns
    -------
    an : ndarray
        Scatttering coefficient M function
    bn : ndarray
        Scattering coefficient N function
    '''
    # first check inputs and arrange them in np arrays
    if check_inputs:
        lam, Nh, Np, D = _check_mie_inputs(lam,N_host,Np_shells,D)
    
    m = Np/Nh                       # sphere layers
    R = D/2                         # particle's inner radius
    kh = 2*pi*Nh/lam                # wavector in the host
    x = _np.tensordot(kh,R,axes=0)   # size parameter
    m = m.transpose()

    # determine nmax 
    if nmax is None :
        y = max(x[-1,:]) # largest size parameter of outer layer
        # define nmax according to B.R Johnson (1996)
        nmax = int(_np.round(_np.abs(y) + 4*_np.abs(y)**(1/3) + 2))

    # Preallocate outputs
    an = _np.zeros((len(lam), nmax), dtype=complex)
    bn = _np.zeros((len(lam), nmax), dtype=complex)
    for i in range(len(lam)):
        an[i,:], bn[i, :], *_ = _get_coated_coefficients(m[i, :], x[i, :], nmax)
    
    return an.reshape(-1, nmax), bn.reshape(-1, nmax)

def _pi_tau_1n(theta, nmax):
    """
    Compute the scalar tesseral function π_1n(θ) and τ_1n(θ)
    The arrays start with n = 1

    Adapted from the miepython library: https://github.com/scottprahl/miepython
    Original Author: Scott Prahl
    Modifications by: Francisco Ramírez (2025)
    
    Parameters:
        theta (ndarray): Polar angle θ in radians.
        nmax (int): Max degree of the associated Legendre polynomial.
        
    Returns:
        ndarray: π_1n(θ) = P_n^1(cos𝜃) / sin𝜃.
        ndarray: τ_1n(θ) = d/d𝜃 P_n^1(cos𝜃).
    """
    mu = _np.cos(theta)  # x = cos(θ)

    pi  = _np.zeros((nmax, len(mu)))
    tau = _np.zeros((nmax, len(mu)))
    
    pi_nm2 = 0
    pi[0] = _np.ones_like(mu)
    
    for n in range(1, nmax):
        tau[n - 1] =            n * mu * pi[n - 1] - (n + 1) * pi_nm2
        temp = pi[n - 1]
        pi [n    ] = ((2 * n + 1) * mu * temp        - (n + 1) * pi_nm2) / n
        pi_nm2 = temp
        
    return pi, tau

def scatter_amplitude(theta: _Union[float, _np.ndarray], 
                      lam: _Union[float, _np.ndarray], 
                      N_host: _Union[float, _np.ndarray], 
                      Np_shells: _Union[float, _np.ndarray], 
                      D: _Union[float, _np.ndarray], 
                      nmax: int = None,
                      check_inputs: bool = True):
    """
    Calculate the elements S1 (S11) and S2 (S22) of the scattering matrix for spheres.
    * For spheres S12 = S21 = 0

    The amplitude functions have been normalized so that when integrated
    over all 4*pi solid angles, the integral will be qext*pi*x**2.

    Adapted from the miepython library: https://github.com/scottprahl/miepython
    Original Author: Scott Prahl
    Modifications by: Francisco Ramírez (2025)

    Parameters:
        theta (ndarray or float): Scattering angle (radians)

        lam (ndarray or float): wavelengtgh (microns)
        
        N_host (ndarray or float): Complex refractive index of host. If 
                                   ndarray, len = lam
        
        Np_shells (float, 1darray or list): Complex refractive index of each 
                                            shell layer. The number of elements 
                                            must be equal to len(D). 
            Options are:
            float:   solid sphere and constant refractive index
            1darray: solid sphere and spectral refractive index (len = lam)
            list:    multilayered sphere (with both constant or spectral refractive indexes)
        
        D (float or list): Outter diameter of each shell's layer (microns). 
            Options are:
            float: solid sphere
            list:  multilayered sphere

        nmax (int, optional): Number of mie scattering coefficients. Default None

    Returns:
        S1, S2: the scattering amplitudes at each angle mu [sr**(-0.5)]
    """
    # first check inputs and arrange them in np arrays
    if check_inputs:
        lam, Nh, Np, D = _check_mie_inputs(lam,N_host,Np_shells,D)
    
    # convert input variables to array
    if _np.isscalar(theta) : theta = _np.array([theta,])

    # Extract mie scattering coefficients
    an, bn = scatter_coeffients(lam,N_host,Np_shells,D, nmax, check_inputs=False)
    nmax = an.shape[1]

    # get pi and tau angular functions
    pi, tau = _pi_tau_1n(theta, nmax)

    # set scale for summation
    n = _np.arange(1, nmax + 1)
    scale = (2 * n + 1) / ((n + 1) * n)

    mu = _np.cos(theta)

    # compute S1 and S2
    S1 = _np.zeros((len(mu), len(lam)), dtype=_np.complex128)
    S2 = _np.zeros((len(mu), len(lam)), dtype=_np.complex128)
    for k in range(len(mu)):
        S1[k] = _np.dot(scale* pi[:,k],an.T) + _np.dot(scale*tau[:,k],bn.T)
        S2[k] = _np.dot(scale*tau[:,k],an.T) + _np.dot(scale* pi[:,k],bn.T)

    return S1, S2

def scatter_stokes(theta: _Union[float, _np.ndarray], 
                   lam: _Union[float, _np.ndarray], 
                   N_host: _Union[float, _np.ndarray], 
                   Np_shells: _Union[float, _np.ndarray], 
                   D: _Union[float, _np.ndarray], 
                   nmax: int = None,
                   check_inputs: bool = True):
    """
    Calculate the Stokes parameters S11, S12, S33 and S34 of a sphere. 

    Parameters:
        theta (ndarray or float): Scattering angle (radians)

        lam (ndarray or float): wavelengtgh (microns)
        
        N_host (ndarray or float): Complex refractive index of host. If 
                                   ndarray, len(N_host) == len(lam)
        
        Np_shells (float, 1darray or list): Complex refractive index of each 
                                            shell layer. Np_shells.shape[1] == len(D). 
            Options are:
            float:   solid sphere and constant refractive index
            1darray: solid sphere and spectral refractive index (len = lam)
            list:    multilayered sphere (with both constant or spectral refractive indexes)
        
        D (float or list): Outter diameter of each shell's layer (microns). 
            Options are:
            float: solid sphere
            list:  multilayered sphere

        nmax (int, optional): Number of mie scattering coefficients. Default None

        as_ndarray (bool): True if user wants the output as ndarray. Otherwise, 
        the output is a pd.DataFrame. Default False

    Returns:
        phase_fun: the scattering phase function (as pd.DataFrame or ndarray)
    """

    # Organize D format
    if check_inputs:
        lam, Nh, Np, D = _check_mie_inputs(lam,N_host,Np_shells,D)
    
    # convert input variables to array
    if _np.isscalar(theta) : theta = _np.array([theta,])
    
    # Get scattering amplitude elements S1 and S2
    s1, s2 = scatter_amplitude(theta, lam,N_host,Np_shells,D, nmax, check_inputs = False)

    # Compute stokes parameters
    S11 =1/2*(_np.abs(s1)**2 + _np.abs(s2)**2)
    S12 =1/2*(_np.abs(s1)**2 - _np.abs(s2)**2)
    S33 =1/2*(s2.conj()*s1 + s2*s1.conj())
    S34 =1*2*(s2.conj()*s1 - s2*s1.conj())

    return S11, S12, S33, S34

def phase_scatt(theta: _Union[float, _np.ndarray], 
                   lam: _Union[float, _np.ndarray], 
                   N_host: _Union[float, _np.ndarray], 
                   Np_shells: _Union[float, _np.ndarray], 
                   D: _Union[float, _np.ndarray], 
                   nmax: int = None, 
                   as_ndarray: bool = False,
                   check_inputs: bool = True):
    """
    Calculate the scattering phase function. The intensity is normalized 
    such that the integral is equal to qsca

    Adapted from the miepython library: https://github.com/scottprahl/miepython
    Original Author: Scott Prahl
    Modifications by: Francisco Ramírez (2025)

    Parameters:
        theta (ndarray or float): Scattering angle (radians)

        lam (ndarray or float): wavelengtgh (microns)
        
        N_host (ndarray or float): Complex refractive index of host. If 
                                   ndarray, len(N_host) == len(lam)
        
        Np_shells (float, 1darray or list): Complex refractive index of each 
                                            shell layer. Np_shells.shape[1] == len(D). 
            Options are:
            float:   solid sphere and constant refractive index
            1darray: solid sphere and spectral refractive index (len = lam)
            list:    multilayered sphere (with both constant or spectral refractive indexes)
        
        D (float or list): Outter diameter of each shell's layer (microns). 
            Options are:
            float: solid sphere
            list:  multilayered sphere

        nmax (int, optional): Number of mie scattering coefficients. Default None

        as_ndarray (bool): True if user wants the output as ndarray. Otherwise, 
        the output is a pd.DataFrame. Default False

    Returns:
        phase_fun: the scattering phase function (as pd.DataFrame or ndarray)
    """
    # Organize D format
    if check_inputs:
        lam, Nh, Np, D = _check_mie_inputs(lam,N_host,Np_shells,D)
    
    # convert input variables to array
    if _np.isscalar(theta) : theta = _np.array([theta,])
    
    # Get scattering amplitude elements S1 and S2
    s1, s2 = scatter_amplitude(theta, lam,N_host,Np_shells,D, nmax, check_inputs = False)

    # Scale factor
    x = _np.pi*Nh.real*D[-1]/lam
    scale_factor = _np.pi*x**2

    # Compute phase function
    phase_fun = 1/scale_factor*(_np.abs(s1)**2 + _np.abs(s2)**2)/2

    # return phase function as ndarray
    if as_ndarray: return phase_fun

    # if not convert phase function to dataframe
    df_phase_fun = pd.DataFrame(data=phase_fun, 
                            index=_np.degrees(theta), 
                            columns=lam)

    return df_phase_fun

def phase_scatt_HG(theta, lam, gcos, qsca = 1, as_ndarray = False):
    """
    Compute the Heyney-Greenstein phase function

    Parameters
        theta : float or ndarray
            Scatttering angle (radians)
        gcos : float or ndarray
            Asymmetry parameter
        qsca: float or ndarray (optional)
            Scattering efficiency. If 1, then integral of phase function = 1.
            Default 1

    Return
        p_theta_HG: float or ndarray
            Phase function
    """
    if _np.isscalar(theta): theta = _np.array([theta])
    if _np.isscalar(gcos): theta = _np.array([gcos])
    if not _np.isscalar(qsca) and (len(qsca) != len(gcos)): 
        raise ValueError("qsca and gcos must be of same size.")

    gg, tt = _np.meshgrid(gcos, theta)

    p_theta_HG = 1/(4*_np.pi)*(1 - gg**2)/(1 + gg**2 - 2*gg*_np.cos(tt))**(3/2)

    p_theta_HG = qsca*p_theta_HG

    # return phase function as ndarray
    if as_ndarray: return p_theta_HG

    # if not convert phase function to dataframe
    df_phase_fun = pd.DataFrame(data=p_theta_HG, 
                            index=_np.degrees(theta), 
                            columns=lam)

    return df_phase_fun
    
def scatter_from_phase_function(phase_fun):
    """
    Compute Qsca and <cos theta> from a DataFrame whose rows are labeled
    with scattering angles in degrees and columns with wavelengths.
    
    Parameters
    ----------
    phase_fun : pd.DataFrame
        Phase function. Row index must be theta in degrees from 0 to over 180.
        Columns correspond to different wavelengths.

    Returns
    -------
    qsca : ndarray
        Scattering efficiency for each column.
        
    gcos : ndarray
        Asymmetry parameter for each column.
    """
    # Organize D format
    # _, Nh, _, D = _check_mie_inputs(N_host = N_host, D = D)

    # Step 1: Sort index by angle
    phase_fun = phase_fun.sort_index()
    lam = phase_fun.columns.to_numpy()  # wavelength (um)

    # Step 2: Subset to angles [0°, 180°]
    subset = phase_fun.loc[(phase_fun.index >= 0) & (phase_fun.index <= 180)]

    # Step 3: Validation
    theta = subset.index.to_numpy()
    if len(theta) < 2:
        raise ValueError("Not enough angle samples between 0 and 180 degrees.")

    if not _np.isclose(theta[0], 0, atol=3) or not _np.isclose(theta[-1], 180, atol=3):
        raise ValueError("Selected theta range must span from 0 to 180 degrees.")

    if not _np.all(_np.diff(theta) > 0):
        raise ValueError("Theta values must be strictly increasing — no duplicates allowed.")

    mu = _np.cos(_np.radians(theta))

    # Sort phase function and mu in ascending order
    p_theta = subset.values[_np.argsort(mu)]
    mu.sort()

    # compute scattering efficiency and asymmetry parameter
    qsca = 2 * _np.pi * _np.trapz(p_theta, mu, axis=0)
    gcos = 2 * _np.pi * _np.trapz(mu*p_theta.T, mu, axis=1)/qsca

    # sanitize NaNs/infs if any wavelength has vanishing scattering
    mask_bad = ~_np.isfinite(qsca) | (qsca <= 0)
    if _np.any(mask_bad):
        gcos[mask_bad] = 0.0
        qsca[mask_bad] = 0.0

    return qsca, gcos

def _mono_percus_yevick(fv, q, D):
    """
    Compute the Percus-Yevick structure factor S(q) for monodispersed 
    hard-sphere systems.

    References: Kinning, D. J., & Thomas, E. L. (1984). 
                Hard-Sphere Interactions between Spherical Domains in Diblock Copolymers. 
                Macromolecules, 17(9), 1712–1718.

    Parameters:
    -----------
    fv : float
        Volume fraction (phi) of the spheres.
    q : float
        Magnitude of the scattering vector.
    D : float 
        Diameter of the sphere.

    Returns:
    --------
    S_q : float
        Structure factor evaluated at q.
    """
    if not isinstance(D, float) and not isinstance(D, int):
        raise ValueError("For monodisperse case, D must be a float or int.")
    
    R = D / 2
    x = 2 * q * R  # Scattering variable as defined by Kinning & Thomas

    # Coefficients from Eq. (17)
    α = (1 + 2 * fv)**2 / (1 - fv)**4
    β = -6 * fv * (1 + fv / 2)**2 / (1 - fv)**4
    γ = 0.5 * fv * (1 + 2 * fv)**2 / (1 - fv)**4

    # G(A) from Eq. (21)
    term1 = α / x**2 * (_np.sin(x) - x * _np.cos(x))
    term2 = β / x**3 * (2 * x * _np.sin(x) + (2 - x**2) * _np.cos(x) - 2)
    term3 = γ / x**5 * (-x**4 * _np.cos(x) +
                        4 * ((3 * x**2 - 6) * _np.cos(x) +
                                (x**3 - 6 * x) * _np.sin(x) + 6))
    G_kt = term1 + term2 + term3

    # Structure factor from Eq. (20)
    S_q = 1 / (1 + 24 * fv * G_kt / x)
    return S_q

def _poly_percus_yevick(fv, qq, D, nD):
    """
    Compute the Percus-Yevick structure factor S(q) for polydisperse 
    hard-sphere systems.

    References: Botet, R., Kwok, R., & Cabane, B. (2020). 
                Percus–Yevick structure factors made simple. 
                Journal of Applied Crystallography, 53(6), 1526–1534.

    Parameters:
    -----------
    fv : float
        Volume fraction (phi) of the spheres.
    qq : ndarray
        Magnitude of the scattering vector.
    D : ndarray
        Diameter of the spheres
    nD : _np.ndarray or None
        Probability distribution over D (same length as D). If None, assumes monodisperse.

    Returns:
    --------
    S_q : float
        Structure factor evaluated at q.
    """
    if not isinstance(D, _np.ndarray) or not isinstance(nD, _np.ndarray):
        raise ValueError("D and nD must be numpy arrays in the polydisperse case.")
        
    if D.shape != nD.shape:
        raise ValueError("D and nD must have the same shape.")

    R = D / 2

    # Weighted average over size distribution
    average = lambda f: _np.trapz(f * nD, R, axis = 1)  

    # if fv > 0.5, compute structure factor for voids
    # "complementary PY hard-sphere approach"
    if fv > 0.5:
        R = (1 - fv)/fv*R
        fv = 1 - fv

    S_q = _np.zeros_like(qq)
    for i in range(qq.shape[0]):
        q = _np.meshgrid(R, qq[i,:])[1]
        
        x = q * R  # Scattering vector scaled by radius
        
        # Psi is an auxiliary prefactor: psi = 3*phi / (1 - phi)
        psi = 3 * fv / (1 - fv)
    
        # Trigonometric building blocks for structure factor (Botet et al., Eqs. 8–13)
        Fcs = _np.cos(x) + x * _np.sin(x)  # cos(x) + x·sin(x)
        Fsc = _np.sin(x) - x * _np.cos(x)  # sin(x) - x·cos(x)
    
        # Botet et al. expressions for b, c, d, e, f, g
        b = psi * average(Fcs * Fsc) / average(x**3)
        c = psi * average(Fsc**2) / average(x**3)
        d = 1 + psi * average(x**2 * _np.sin(x) * _np.cos(x)) / average(x**3)
        e = psi * average(x**2 * _np.sin(x)**2) / average(x**3)
        f = psi * average(x * _np.sin(x) * Fsc) / average(x**3)
        g = - psi * average(x * _np.cos(x) * Fsc) / average(x**3)
        # print(c)
        
        # Auxiliary variables for S(q)
        denom = d**2 + e**2
        X = 1 + b + (2 * e * f * g + d * (f**2 - g**2)) / denom
        Y = c + (2 * d * f * g - e * (f**2 - g**2)) / denom
    
        # Final expression of S(q) (Eq. 4)
        S_q[i,:] = (Y / c) / (X**2 + Y**2)
        
    return S_q

def structure_factor_PY(theta: _Union[float, _np.ndarray], 
                        lam: _Union[float, _np.ndarray], 
                        Nh: _Union[float, _np.ndarray], 
                        D: _Union[float, _np.ndarray], 
                        fv: float, 
                        nD: _np.ndarray = None, 
                        check_inputs: bool = True):
    """
    Compute the Percus-Yevick structure factor S(q) for hard-sphere systems,
    for both monodisperse and polydisperse cases.

    Parameters:
    -----------
    fv : float
        Volume fraction (phi) of the spheres.
    lam : float or ndarray
        Wavelength range (um)
    Nh: float or ndarray
        Refractive index of host. If ndarray, len(Nh) == len(lam)
    D : float or _np.ndarray
        Diameter of the spheres. Use float for monodisperse, or array for polydisperse.
    nD : _np.ndarray or None
        Probability distribution over D (same length as D). If None, assumes monodisperse.

    Returns:
    --------
    S_q : float
        Structure factor evaluated at q.

    Raises:
    -------
    ValueError
        If inputs are inconsistent or invalid.
    """    
    if isinstance(theta, float): theta = _np.array([theta])
    
    if check_inputs:
        lam, Nh, _, _ = _check_mie_inputs(lam, Nh)
    
    # compute scattering vector (q = 2k0*sin(theta/2))
    k0 = 2*_np.pi*Nh.real/lam
    q = _np.outer(2*k0, _np.sin(theta/2))

    q[q < 0.1] = 0.1  # Found overflow for q < 0.1
    
    if nD is None:
        S_q = _mono_percus_yevick(fv, q, D).T

    else:
        S_q = _poly_percus_yevick(fv, q, D, nD).T
    
    return S_q

def phase_scatt_dense(theta, lam, N_host, Np, D, fv, nD=None, *, 
                      nmax=None, as_ndarray=False, effective_medium=True):
    """
    Calculate the scattering phase function for multiple hard-spheres under unpolarized light. 
    The intensity is normalized such that the integral is equal to qsca

    Parameters:
        theta : ndarray or float
            Scattering angle (radians)
        lam : ndarray or float 
            Wavelengtgh (microns)
        N_host : ndarray or float 
            Complex refractive index of host. If ndarray, len(N_host) == len(lam)
        Np : float or ndarray
            Complex refractive index of the sphere. If ndarray, len(Np) == len(lam). 
        D : float or _np.ndarray
            Diameter of the spheres. Use float for monodisperse, or array for polydisperse.
        fv: float
            Filling fraction
        nD: ndarray
            Diameter density distribution. len(nD) == len(D)
        nmax : int (optional)
            Number of mie scattering coefficients. Default None
        as_ndarray : bool (optional)
            True if user wants the output as ndarray. Otherwise, the output is a pd.DataFrame. 
            Default False
        effective_medium : bool (optional)
            If True, compute the effective refractive index of the host using Bruggeman EMT.
            Default True

    Returns:
        phase_fun: the scattering phase function (as pd.DataFrame or ndarray)
    """
    # Input checks
    if nD is not None:
        if not isinstance(D, _np.ndarray):
            raise ValueError("For polydisperse case, D must be a numpy array.")
        if not isinstance(nD, _np.ndarray):
            raise ValueError("nD must be a numpy array if provided.")
        if D.shape != nD.shape:
            raise ValueError("D and nD must have the same shape.")
        if _np.any(nD < 0):
            raise ValueError("nD must be non-negative.")
        if not _np.isclose(_np.sum(nD), 1, atol=1e-2):
            nD = nD / _np.sum(nD)  # Normalize if not already
    if not isinstance(effective_medium, bool):
        raise ValueError("effective_medium must be a boolean value.")

    if effective_medium:
        # Compute effective refractive index of host using Bruggeman EMT
        N_host = emt_brugg(fv, Np, N_host)

    # Get form factor
    if nD is None:
        # Monodisperse
        F_theta = phase_scatt(theta, lam, N_host, Np, D, nmax, as_ndarray=True)
    else:
        Ac = _np.pi*(D/2)**2  # cross-sectional area of each diameter

        # Polydisperse: ensemble average over diameter distribution
        F_theta = _np.zeros((len(theta), len(lam)), dtype=float)
        for i, Di in enumerate(D):
            # For each diameter, compute phase function
            F_theta += nD[i] * Ac[i] * phase_scatt(theta, lam, N_host, Np, Di, nmax, as_ndarray=True)
        
        # Normalize by average cross-sectional area
        F_theta /= _np.sum(nD * Ac)

    # Get structure factor
    S_q = structure_factor_PY(theta, lam, N_host, D, fv, nD)

    phase_fun = F_theta * S_q

    # return phase function as ndarray
    if as_ndarray:
        return phase_fun

    # if not convert phase function to dataframe
    df_phase_fun = pd.DataFrame(data=phase_fun, 
                               index=_np.degrees(theta), 
                               columns=lam)

    return df_phase_fun

def poly_sphere_cross_section(
    lam, D_list, p_list, Np, Nh, fv, *,
    n_theta: int = 361,             # dense angular grid for forward peaks
    atol_prob: float = 1e-6,        # tolerance for sum(p)=1
    effective_medium: bool = True,  # whether to compute effective Nh via Bruggeman
):
    """
    Compute size-averaged scattering/absorption cross sections and asymmetry parameter
    for a polydisperse set of hard spheres under the independent-scattering assumption.
    Not valid for metallic spheres or high volume fractions where near-field coupling
    is important.

    Parameters
    ----------
    lam : array-like, shape (nλ,)
        Wavelengths [µm], strictly positive.
    D_list : array-like, shape (nD,)
        Particle diameters [µm], strictly positive.
    p_list : array-like, shape (nD,)
        Number-fraction probabilities for each diameter (Case A). Sum must be 1
        within tolerance; will be renormalized if slightly off.
    Np : float or array-like (nλ,)
        Particle refractive index (can be complex). If array-like, length must equal len(lam).
    Nh : float or array-like (nλ,)
        Host refractive index (can be complex). If array-like, length must equal len(lam).
    fv : float
        Particle volume fraction in (0, 1). Used only to compute an effective medium Nh via
        `nk.emt_brugg(fv, Np, Nh)`.
    n_theta : int, optional
        Number of polar angles for phase function integration (default: 361). Must be >= 5.
    atol_prob : float, optional
        Absolute tolerance for sum(p_list) to be considered 1 (default: 1e-6).
    effective_medium : bool, optional
        Whether to compute an effective host refractive index via Bruggeman EMT (default: True

    Returns
    -------
    csca_av : _np.ndarray, shape (nλ,)
        Size-averaged scattering cross section per particle [µm²].
    cabs_av : _np.ndarray, shape (nλ,)
        Size-averaged absorption cross section per particle [µm²].
    g_av : _np.ndarray, shape (nλ,)
        Size-averaged asymmetry parameter (⟨cosθ⟩).
    """
    # ---------- Input sanitation ----------
    D_list = _np.atleast_1d(_np.asarray(D_list, dtype=float))
    p_list = _np.atleast_1d(_np.asarray(p_list, dtype=float))

    if D_list.ndim != 1 or p_list.ndim != 1:
        raise ValueError("D_list and p_list must be 1D arrays.")
    if D_list.shape != p_list.shape:
        raise ValueError(f"D_list and p_list must have the same length (got {len(D_list)} vs {len(p_list)}).")
    if _np.any(D_list <= 0):
        raise ValueError("All diameters in D_list must be > 0.")
    if not (0 <= fv < 1):
        raise ValueError("fv (volume fraction) must be in [0, 1).")
    if _np.any(_np.real < 0) or _np.any(Nh.real < 0):
        raise ValueError("Refractive indices must have nonnegative real parts.")
    if _np.any(_np.real < _np.imag):
        raise Warning("Method not valid for metallic particles")

    sp = p_list.sum()
    if not _np.isfinite(sp) or sp <= 0:
        raise ValueError("p_list must contain finite nonnegative values and sum to a positive number.")
    if not _np.isclose(sp, 1.0, atol=atol_prob):
        p_list = p_list / sp  # soft-renormalize to 1

    # ---------- Effective medium for host (if your convention is to dress Nh) ----------
    if effective_medium:
        if fv == 0:
            Nh_eff = Nh
        else:
            Nh_eff = emt_brugg(fv, Np, Nh)  # complex array length nλ
    else:
        Nh_eff = Nh

    # ---------- Precompute geometry ----------
    Ac_list = _np.pi * (D_list / 2.0) ** 2          # [µm²]
    # V_list = (4.0/3.0) * _np.pi * (D_list/2.0)**3 # [µm³]  # (unused here)

    # ---------- Absorption: average q_abs * area ----------
    cabs_av = _np.zeros_like(lam, dtype=float)
    for D, p, Ac in zip(D_list, p_list, Ac_list):
        # mie.scatter_efficiency must return arrays shaped (nλ,)
        qext, qsca, _ = scatter_efficiency(lam, Nh_eff, Np, D)
        qabs = qext - qsca
        # sanitize any tiny negative due to numerics
        qabs = _np.where(qabs < 0, 0.0, qabs)
        cabs_av += p * qabs * Ac

    # ---------- Scattering and g: via dense phase function integration ----------
    # Angular grid (dense & includes endpoints)
    n_theta = max(int(n_theta), 5)
    theta = _np.linspace(0.0, _np.pi, n_theta)

    # phase_scatt_dense should return a DataFrame with index=θ° and columns=λ (your earlier design)
    phase_fun_df = phase_scatt_dense(theta, lam, Nh_eff, Np, D_list, fv, p_list, effective_medium=False)

    # Compute Q_sca and g from differential efficiency
    qsca_av, g_av = scatter_from_phase_function(phase_fun_df)

    # Convert Q_sca (efficiency) to cross section via weighted area ⟨A⟩ = Σ p_i A_i
    A_mean = float(_np.sum(p_list * Ac_list))
    csca_av = qsca_av * A_mean

    return csca_av, cabs_av, g_av, phase_fun_df


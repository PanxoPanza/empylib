# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 23:38:11 2021

@author: PanxoPanza
"""
import numpy as np
from numpy import pi, exp, conj, imag, real, sqrt
from scipy.special import jv, yv
import pandas as pd

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
    if np.isscalar(x): x = np.array([x])
    
    n = np.array(range(nmax))
    
    # Get Dn(x) by downwards recurrence
    Dnx = np.zeros((len(x),nmx),dtype=np.complex128)
    for i in reversed(range(1, nmx)):
        # define D_(i+1) (x)
        # if i == nmx-1 : Dip1 = np.zeros(len(x))
        # else :          Dip1 = Dnx[:,i+1]
        
        Dnx[:,i-1] = (i+1)/x - 1/(Dnx[:,i] + (i+1)/x)
        
    # Get Gn(x) by upwards recurrence
    Gnx = np.zeros((len(x),nmx),dtype=np.complex128)
    G0x = 1j*np.ones_like(x)
    i = 0
    Gnx[:,i] = 1/((i+1)/x - G0x) - (i+1)/x
    for i in range(1, nmx):
        # define G_(i-1) (x)
        # if i == 0 : Gim1x = 1j*np.ones(len(x))
        # else : Gim1x = Gnx[:,i-1] 
        
        Gnx[:,i] = 1/((i+1)/x - Gnx[:,i-1]) - (i+1)/x
    
    # Get Rn(x) by upwards recurrence
    Rnx = np.zeros((len(x),len(n)),dtype=np.complex128) 
    for ix in range(len(x)):
        
        # note that 0.5*(1 - exp(-2j*x)) = 0 if x = pi*n
        # I added this clause for those cases
        if imag(x[ix]) == 0 and np.mod(real(x[ix]),pi) == 0:
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
        an = np.zeros(n)
        bn = np.zeros(n)
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
        nmax = int(np.round(np.abs(ka) + 4*np.abs(ka)**(1/3) + 2))
    
    #----------------------------------------------------------------------
    #       Computing an and bn (main part of this code)
    #----------------------------------------------------------------------
    
    mix = m*x               # Ni*k*ri
    mi1 = np.append(m,1)
    mi1x = mi1[1:]*x        # Ni+1*k*ri
    
    # Computation of Dn(z), Gn(z) and Rn(z)
    nmx = int(np.round(max(nmax, max(abs(m*x))) + 16))
    
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
    n = np.array(range(1,nmax+1))
    nu = n+0.5
    phi = np.sqrt(0.5*pi*ka)*jv(nu,ka) # phi(n,ka)
    chi = np.sqrt(0.5*pi*ka)*yv(nu,ka) # chi(n,ka)
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
        nmax = int(np.round(np.abs(y) + 4*np.abs(y)**(1/3) + 2))

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
    n = np.array(range(1,nmax+1))
    
    #------------------------------------------------------------------
    # Extinction efficiency
    #------------------------------------------------------------------
    en = (2*n+1)*imag((- 2j*py*conj(py)*imag(Dy)         \
                       + conj(an)*conj(xy)*py*Dy         \
                       - conj(bn)*conj(xy)*py*conj(Gy)   \
                       + an*xy*conj(py)*Gy               \
                       - bn*xy*conj(py)*conj(Dy))        \
                       /y)
    q = np.sum(en)
    Qext = real(1/real(y)*ft*q)    
    
    #------------------------------------------------------------------
    # Scattering efficiency
    #------------------------------------------------------------------
    en = (2*n+1)*imag((+ np.abs(an*xy)**2*Gy                \
                       - np.abs(bn*xy)**2*conj(Gy)         \
                       )/y)
    q = np.sum(en)
    Qsca = real(1/real(y)*ft*q)
    
    #------------------------------------------------------------------
    # Asymmetry parameter
    #------------------------------------------------------------------
    anp1 = np.zeros(nmax,dtype=np.complex128)
    bnp1 = np.zeros(nmax,dtype=np.complex128)
    anp1[:nmax-1] = an[1:] # a(n+1) coefficient
    bnp1[:nmax-1] = bn[1:] # a(n+1) coefficient
    
    asy1 = n*(n + 2)/(n + 1)*(an*conj(anp1)+ bn*conj(bnp1)) \
         + (2*n + 1)/(n*(n + 1))*real(an*conj(bn))
    
    asy2 = (2*n+1)*(an*conj(an) + bn*conj(bn))
    Asym = real(2*np.sum(asy1)/np.sum(asy2))
    
    #------------------------------------------------------------------
    # Backward scattering (not valid for absorbing host media)
    #------------------------------------------------------------------
    f = (2*n+1)*(-1)**n*(an - bn)
    q = np.sum(f)
    Qb = real(q*conj(q)/y**2)
    
    #------------------------------------------------------------------
    # Forward scattering (not valid for absorbing host media)
    #------------------------------------------------------------------
    f = (2*n+1)*(an + bn)
    q = np.sum(f)
    Qf = real(q*conj(q)/y**2)
    
    #------------------------------------------------------------------
    # Condition outputs to avoid unphysical results
    #------------------------------------------------------------------
    if Qsca < 0: Qsca = 0
    if Qext < Qsca: Qext = Qsca
    if Asym < -1: Asym = -1
    if Asym > +1: Asym = +1

    return Qext, Qsca, Asym, Qb, Qf

def _check_mie_inputs(lam=None,N_host=None,Np_shells=None,D=None):
    '''
    Ckeck and organize mie inputs before running any simulations
    
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

    Returns
    -------
    lam : 1darray
        wavelength range

    Nh : 1darray
        refractive index of host 

    Np : ndarray
        refractive index of shell layers

    D  : 1darray
        Diameters of shell layers

    '''
    
    # convert input variables to list
    if lam is not None:
        if np.isscalar(lam) : lam = np.array([lam,])

    # Verify D is float or list
    if D is not None:
        #   1. solid sphere
        if np.isscalar(D) : D = [D,]
        #   2. multilayered sphere
        else:
            assert isinstance(D, list), 'diameter of shell layers must be on a list format'
        
        # convert list to ndarrays
        D = np.array(D)

    # Verify Np_shells is float, 1darray or list 
    if Np_shells is not None:
        #   1.solid sphere constant refractive index
        if np.isscalar(Np_shells): 
            Np_shells = [Np_shells,]
        #   2.solid sphere spectral refractive index
        elif isinstance(Np_shells, np.ndarray) and Np_shells.ndim ==1:
            Np_shells = [Np_shells,]
        #   3. multilayered sphere
        else:
            assert isinstance(Np_shells, list), 'refractive index of shell layers must be on a list format'
    
    # if multilayered sphere, check refractive index and D match in length
    if Np_shells is not None and D is not None:
        assert len(D) == len(Np_shells), 'number of layers in D and Np_shells must be the same'

        # analize Np_shells and rearrange to ndarray if float
        Np = []
        for Ni in Np_shells:
            if np.isscalar(Ni):            # convert to ndarray if float
                Ni = np.ones(len(lam))*Ni
            else: 
                assert len(Ni) == len(lam), 'Np_layers must either float or size len(lam)'
            Np.append(Ni.astype(complex))
        Np = np.array(Np).reshape(len(D),len(lam))

        # sort layers from inner to outer shell
        idx = np.argsort(D)
        D = D[idx]
        Np_shells = Np[idx,:]

    # analyze N_host and rearrange to ndarray if float
    if N_host is not None and lam is not None:
        if np.isscalar(N_host):             # convert to ndarray if float
            N_host = np.ones(len(lam), dtype = complex)*N_host
        else: 
            assert len(N_host) == len(lam), 'N_host must either float or size len(lam)'
            N_host = np.copy(N_host.astype(complex))

    return lam, N_host, Np_shells, D 

def scatter_efficiency(lam,N_host,Np_shells,D,nmax=None):
    
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
    lam, Nh, Np, D = _check_mie_inputs(lam,N_host,Np_shells,D)

    m = Np/Nh.real                  # sphere layers
    R = D/2                         # particle's inner radius
    kh = 2*pi*Nh.real/lam           # wavector in the host
    x = np.tensordot(kh,R,axes=0)   # size parameter
    m = m.transpose()
    
    get_cross_section = np.vectorize(_cross_section_at_lam, 
                                signature = '(n),(n),() -> (),(),(),(),()')
        
    # outputs: qext, qsca, gcos
    return get_cross_section(m, x, nmax)[:3] 
    
def scatter_coeffients(lam,N_host,Np_shells,D, nmax = None):
    
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
    lam, Nh, Np, D = _check_mie_inputs(lam,N_host,Np_shells,D)
    
    m = Np/Nh                       # sphere layers
    R = D/2                         # particle's inner radius
    kh = 2*pi*Nh/lam                # wavector in the host
    x = np.tensordot(kh,R,axes=0)   # size parameter
    m = m.transpose()

    # determine nmax 
    if nmax is None :
        y = max(x[-1,:]) # largest size parameter of outer layer
        # define nmax according to B.R Johnson (1996)
        nmax = int(np.round(np.abs(y) + 4*np.abs(y)**(1/3) + 2))

    get_coefficients = np.vectorize(_get_coated_coefficients,
                signature = '(n), (n), () -> (m), (m), (m), (m), (m), (m)')

    # outputs an and bn
    an, bn = get_coefficients(m, x, nmax)[:2]
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
    mu = np.cos(theta)  # x = cos(θ)

    pi  = np.zeros((nmax, len(mu)))
    tau = np.zeros((nmax, len(mu)))
    
    pi_nm2 = 0
    pi[0] = np.ones_like(mu)
    
    for n in range(1, nmax):
        tau[n - 1] =            n * mu * pi[n - 1] - (n + 1) * pi_nm2
        temp = pi[n - 1]
        pi [n    ] = ((2 * n + 1) * mu * temp        - (n + 1) * pi_nm2) / n
        pi_nm2 = temp
        
    return pi, tau

def scatter_amplitude(theta, lam,N_host,Np_shells,D, nmax = None):
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
    # convert input variables to array
    if np.isscalar(theta) : theta = np.array([theta,])
    if np.isscalar(lam) : theta = np.array([lam,])

    # Extract mie scattering coefficients
    an, bn = scatter_coeffients(lam,N_host,Np_shells,D, nmax)
    nmax = an.shape[1]

    # get pi and tau angular functions
    pi, tau = _pi_tau_1n(theta, nmax)

    # set scale for sumation
    n = np.arange(1, nmax + 1)
    scale = (2 * n + 1) / ((n + 1) * n)

    mu = np.cos(theta)

    # compute S1 and S2
    S1 = np.zeros((len(mu), len(lam)), dtype=np.complex128)
    S2 = np.zeros((len(mu), len(lam)), dtype=np.complex128)
    for k in range(len(mu)):
        S1[k] = np.dot(scale* pi[:,k],an.T) + np.dot(scale*tau[:,k],bn.T)
        S2[k] = np.dot(scale*tau[:,k],an.T) + np.dot(scale* pi[:,k],bn.T)

    return S1, S2

def scatter_stokes(theta, lam,N_host,Np_shells,D, nmax = None, as_ndarray = False):
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
    # Get scattering amplitude elements S1 and S2
    s1, s2 = scatter_amplitude(theta, lam,N_host,Np_shells,D, nmax)

    # Organize D format
    _, Nh, _, D = _check_mie_inputs(lam = lam, N_host = N_host, D = D)

    # Compute stokes parameters
    S11 =1/2*(np.abs(s1)**2 + np.abs(s2)**2)
    S12 =1/2*(np.abs(s1)**2 - np.abs(s2)**2)
    S33 =1/2*(s2.conj()*s1 + s2*s1.conj())
    S34 =1*2*(s2.conj()*s1 - s2*s1.conj())

    return S11, S12, S33, S34

def phase_scatt(theta, lam,N_host,Np_shells,D, nmax = None, as_ndarray = False):
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
    # Get scattering amplitude elements S1 and S2
    s1, s2 = scatter_amplitude(theta, lam,N_host,Np_shells,D, nmax)

    # Organize D format
    _, Nh, _, D = _check_mie_inputs(lam = lam, N_host = N_host, D = D)

    # Scale factor
    x = np.pi/lam*D[-1]/Nh.real
    scale_factor = np.pi*x**2

    # Compute phase function
    phase_fun = 1/scale_factor*(np.abs(s1)**2 + np.abs(s2)**2)/2

    # return phase function as ndarray
    if as_ndarray: return phase_fun

    # if not convert phase function to dataframe
    df_phase_fun = pd.DataFrame(data=phase_fun, 
                            index=np.degrees(theta), 
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
    if np.isscalar(theta): theta = np.array([theta])
    if np.isscalar(gcos): theta = np.array([gcos])
    if not np.isscalar(qsca) and (len(qsca) != len(gcos)): 
        raise ValueError("qsca and gcos must be of same size.")

    gg, tt = np.meshgrid(gcos, theta)

    p_theta_HG = 1/(4*np.pi)*(1 - gg**2)/(1 + gg**2 - 2*gg*np.cos(tt))**(3/2)

    p_theta_HG = qsca*p_theta_HG

    # return phase function as ndarray
    if as_ndarray: return p_theta_HG

    # if not convert phase function to dataframe
    df_phase_fun = pd.DataFrame(data=p_theta_HG, 
                            index=np.degrees(theta), 
                            columns=lam)

    return df_phase_fun
    
def scatter_from_phase_function(phase_fun):
    """
    Compute Qsca and <cos theta> from a DataFrame whose rows are labeled
    with scattering angles in degrees and columns with wavelengths.
    
    Parameters
    ----------
    D : float or list
        Outter diameter of each shell's layer (microns). Options are:
            float: solid sphere
            list:  multilayered sphere

    N_host (ndarray or float): Complex refractive index of host. If 
                    ndarray, len = lam

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
    # x = np.pi/lam*D[-1]/Nh              # size parameter

    # Step 2: Subset to angles [0°, 180°]
    subset = phase_fun.loc[(phase_fun.index >= 0) & (phase_fun.index <= 180)]

    # Step 3: Validation
    theta = subset.index.to_numpy()
    if len(theta) < 2:
        raise ValueError("Not enough angle samples between 0 and 180 degrees.")

    if not np.isclose(theta[0], 0, atol=3) or not np.isclose(theta[-1], 180, atol=3):
        raise ValueError("Selected theta range must span from 0 to 180 degrees.")

    if not np.all(np.diff(theta) > 0):
        raise ValueError("Theta values must be strictly increasing — no duplicates allowed.")

    mu = np.cos(np.radians(theta))

    # Sort phase function and mu in ascending order
    p_theta = subset.values[np.argsort(mu)]
    mu.sort()

    # compute scattering efficiency and asymmetry parameter
    qsca = 2 * np.pi * np.trapz(p_theta, mu, axis=0)
    gcos = 2 * np.pi * np.trapz(mu*p_theta.T, mu, axis=1)/qsca
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
    term1 = α / x**2 * (np.sin(x) - x * np.cos(x))
    term2 = β / x**3 * (2 * x * np.sin(x) + (2 - x**2) * np.cos(x) - 2)
    term3 = γ / x**5 * (-x**4 * np.cos(x) +
                        4 * ((3 * x**2 - 6) * np.cos(x) +
                                (x**3 - 6 * x) * np.sin(x) + 6))
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
    nD : np.ndarray or None
        Probability distribution over D (same length as D). If None, assumes monodisperse.

    Returns:
    --------
    S_q : float
        Structure factor evaluated at q.
    """
    if not isinstance(D, np.ndarray) or not isinstance(nD, np.ndarray):
        raise ValueError("D and nD must be numpy arrays in the polydisperse case.")
        
    if D.shape != nD.shape:
        raise ValueError("D and nD must have the same shape.")

    R = D / 2

    # Weighted average over size distribution
    average = lambda f: np.trapz(f * nD, R, axis = 1)  

    # if fv > 0.5, compute structure factor for voids
    # "complementary PY hard-sphere approach"
    if fv > 0.5:
        R = (1 - fv)/fv*R
        fv = 1 - fv

    S_q = np.zeros_like(qq)
    for i in range(qq.shape[0]):
        q = np.meshgrid(R, qq[i,:])[1]
        
        x = q * R  # Scattering vector scaled by radius
        
        # Psi is an auxiliary prefactor: psi = 3*phi / (1 - phi)
        psi = 3 * fv / (1 - fv)
    
        # Trigonometric building blocks for structure factor (Botet et al., Eqs. 8–13)
        Fcs = np.cos(x) + x * np.sin(x)  # cos(x) + x·sin(x)
        Fsc = np.sin(x) - x * np.cos(x)  # sin(x) - x·cos(x)
    
        # Botet et al. expressions for b, c, d, e, f, g
        b = psi * average(Fcs * Fsc) / average(x**3)
        c = psi * average(Fsc**2) / average(x**3)
        d = 1 + psi * average(x**2 * np.sin(x) * np.cos(x)) / average(x**3)
        e = psi * average(x**2 * np.sin(x)**2) / average(x**3)
        f = psi * average(x * np.sin(x) * Fsc) / average(x**3)
        g = - psi * average(x * np.cos(x) * Fsc) / average(x**3)
        # print(c)
        
        # Auxiliary variables for S(q)
        denom = d**2 + e**2
        X = 1 + b + (2 * e * f * g + d * (f**2 - g**2)) / denom
        Y = c + (2 * d * f * g - e * (f**2 - g**2)) / denom
    
        # Final expression of S(q) (Eq. 4)
        S_q[i,:] = (Y / c) / (X**2 + Y**2)
        
    return S_q

def structure_factor_PY(theta, lam, Nh, D, fv, nD=None):
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
    D : float or np.ndarray
        Diameter of the spheres. Use float for monodisperse, or array for polydisperse.
    nD : np.ndarray or None
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
    if isinstance(theta, float): theta = np.array([theta])
    
    lam, Nh, _, _ = _check_mie_inputs(lam, Nh)
    
    # compute scattering vector (q = 2k0*sin(theta/2))
    k0 = 2*np.pi*Nh.real/lam
    q = np.outer(2*k0, np.sin(theta/2))

    q[q < 0.1] = 0.1  # Found overflow for q < 0.1
    
    if nD is None:
        S_q = _mono_percus_yevick(fv, q, D).T

    else:
        S_q = _poly_percus_yevick(fv, q, D, nD).T
    
    return S_q

def phase_scatt_dense(theta, lam, N_host, Np, D, fv, nD = None, nmax = None, as_ndarray = False):
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
        
        D : float or np.ndarray
            Diameter of the spheres. Use float for monodisperse, or array for polydisperse.
        
        fv: float
            Filling fraction
        
        nD: ndarray
            Diameter density distribution. len(nD) == D

        nmax : int (optional)
            Number of mie scattering coefficients. Default None

        as_ndarray : bool (optional)
            True if user wants the output as ndarray. Otherwise, the output is a pd.DataFrame. 
            Default False

    Returns:
        phase_fun: the scattering phase function (as pd.DataFrame or ndarray)
    """
    
    # Get form factor
    F_theta = phase_scatt(theta, lam, N_host, Np ,D, nmax, as_ndarray = True)
    
    # Get structure factor
    S_q = structure_factor_PY(theta, lam, N_host, D, fv, nD)

    phase_fun = F_theta*S_q

    # return phase function as ndarray
    if as_ndarray: return phase_fun

    # if not convert phase function to dataframe
    df_phase_fun = pd.DataFrame(data=phase_fun, 
                            index=np.degrees(theta), 
                            columns=lam)

    return df_phase_fun


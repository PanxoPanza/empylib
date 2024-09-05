# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 23:38:11 2021

@author: PanxoPanza
"""
import numpy as np
from numpy import pi, exp, conj, imag, real, sqrt
from scipy.special import jv, yv

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
    Dnx = np.zeros((len(x),nmx),dtype=np.complex_)
    for i in reversed(range(nmx)):
        # define D_(i+1) (x)
        if i == nmx-1 : Dip1 = np.zeros(len(x))
        else :          Dip1 = Dnx[:,i+1]
        
        Dnx[:,i] = (i + 2)/x - 1/(Dip1 + (i + 2)/x)
        
    # Get Gn(x) by upwards recurrence
    Gnx = np.zeros((len(x),nmx),dtype=np.complex_)
    for i in range(nmx):
        # define G_(i-1) (x)
        if i == 0 : Gim1x = 1j*np.ones(len(x))
        else : Gim1x = Gnx[:,i-1] 
        
        Gnx[:,i] = 1/((i + 1)/x - Gim1x) - (i + 1)/x
    
    # Get Rn(x) by upwards recurrence
    Rnx = np.zeros((len(x),len(n)),dtype=np.complex_) 
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
        bn  = np.zeros(n)
    else:
        # get an^i and bn^i
        (an, bn) = _recursive_ab(m[:i],n,
                                Dn[:i-1,:], Gn[:i-1,:], Rn[:i-1,:],
                                Dn1[:i-1,:],Gn1[:i-1,:],Rn1[:i-1,:])
        
        # get Un(mi*kri), Vn(mi, kri)
        Un = (Rn[i-1,:]*Dn[i-1,:] - an*Gn[i-1,:])/(Rn[i-1,:] - an)
        Vn = (Rn[i-1,:]*Dn[i-1,:] - bn*Gn[i-1,:])/(Rn[i-1,:] - bn)
        
        # get an^(i+1), bn^(i+1) by recursion formula
        an = Rn1[i-1,:]*(m[i]/m[i-1]*Un - Dn1[i-1,:])/ \
                        (m[i]/m[i-1]*Un - Gn1[i-1,:])
                      
        bn = Rn1[i-1,:]*(Vn - m[i]/m[i-1]*Dn1[i-1,:])/ \
                        (Vn - m[i]/m[i-1]*Gn1[i-1,:])

    return an, bn
        
def _get_coated_coefficients(m,x, nmax=-1):
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
    if nmax == -1 :
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
    
    return an, bn, phi, Dn1[-1,:], xi, Gn1[-1,:]

def _cross_section_at_lam(m,x,nmax = -1):
    '''
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

    if nmax == -1 :
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
    anp1 = np.zeros(nmax,dtype=np.complex_)
    bnp1 = np.zeros(nmax,dtype=np.complex_)
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
    
    return Qext, Qsca, Asym, Qb, Qf

def _check_mie_inputs(lam,N_host,Np_shells,D):
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
    if np.isscalar(lam) : lam = np.array([lam,])

    # Verify D is float or list
    #   1. solid sphere
    if np.isscalar(D) : D = [D,]
    #   2. multilayered sphere
    else:
        assert isinstance(D, list), 'diameter of shell layers must be on a list format'
    
    # convert list to ndarrays
    D = np.array(D)

    # Verify Np_shells is float, 1darray or list    
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
    assert len(D) == len(Np_shells), 'number of layers in D and Np_shells must be the same'

    # analize Np_shells and rearrange to ndarray if float
    Np = []
    for Ni in Np_shells:
        if np.isscalar(Ni):            # convert to ndarray if float
            Ni = np.ones(len(lam))*Ni
        else: 
            assert len(Ni) == len(lam), 'Np_layers must either float or size len(lam)'
        Np.append(Ni)
    Np = np.array(Np).reshape(len(D),len(lam))

    # sort layers from inner to outer shell
    idx = np.argsort(D)
    D = D[idx]
    Np = Np[idx,:]

    # analyze N_host and rearrange to ndarray if float
    if np.isscalar(N_host):             # convert to ndarray if float
        Nh = np.ones(len(lam))*N_host 
    else: 
        assert len(N_host) == len(lam), 'N_host must either float or size len(lam)'
        Nh = np.copy(N_host)

    return lam, Nh, Np, D 

def scatter_efficiency(lam,N_host,Np_shells,D,nmax=-1):
    
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
        Number of mie scattering coefficients. Default nmax = -1
    
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

    m = Np/Nh                       # sphere layers
    R = D/2                         # particle's inner radius
    kh = 2*pi*Nh/lam                # wavector in the host
    x = np.tensordot(kh,R,axes=0)   # size parameter
    m = m.transpose()
    
    get_cross_section = np.vectorize(_cross_section_at_lam, 
                                signature = '(n),(n),() -> (),(),(),(),()')
        
    # outputs: qext, qsca, gcos
    return get_cross_section(m, x, nmax)[:3] 
    
def scatter_coeffients(lam,N_host,Np_shells,D, nmax = -1):
    
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
        Number of mie scattering coefficients. Default nmax = -1

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
    if nmax == -1 :
        y = max(x[-1,:]) # largest size parameter of outer layer
        # define nmax according to B.R Johnson (1996)
        nmax = int(np.round(np.abs(y) + 4*np.abs(y)**(1/3) + 2))

    get_coefficients = np.vectorize(_get_coated_coefficients,
                signature = '(n), (n), () -> (m), (m), (m), (m), (m), (m)')

    # outputs an and bn
    return get_coefficients(m, x, nmax)[:2]
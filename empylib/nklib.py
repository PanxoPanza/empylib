# -*- coding: utf-8 -*-
"""
Library of tabulated refractive index

Created on Sun Nov  7 17:25:53 2021

@author: PanxoPanza
"""
import os
import platform
import numpy as np 
import pandas as pd
from scipy.integrate import quad
from warnings import warn
from typing import Callable # used to check callable variables
from .utils import _ndarray_check
from pathlib import Path
# import refidx as ri
from .utils import convert_units
import yaml
import requests
from io import StringIO

def get_nkfile(lam, MaterialName, get_from_local_path = False):
    '''
    Reads a tabulated *.nk file and returns an interpolated
    1D numpy array with the complex refractive index
    
    Parameters
    ----------
    lam : ndarray
        Wavelengths to interpolate (um).
    MaterialName : string
        Name of *.nk file

    Returns
    -------
    N : ndarray
        Interpolated complex refractive index
    data: ndarray
        Original tabulated data from file
    '''

    # check if lam is not ndarray
    lam, lam_isfloat = _ndarray_check(lam)   
    
    # retrieve local path
    if get_from_local_path:
        # if function is called locally
        caller_directory = Path(__file__).parent / 'nk_files'
    else :
        # if function is called from working directory (where the function is called)
        caller_directory = Path.cwd()
    
    # Construct the full path of the file
    filename = MaterialName + '.nk'
    file_path = caller_directory / filename   
   
    # check if file exist
    assert file_path.exists(), 'File not found'
    
    # read data as dataframe
    nk_df = pd.read_csv(file_path, comment='#', sep='\s+', header=None, index_col=0)
    
    # check if has n and k data
    assert nk_df.shape[1] == 2, 'wrong file format'

    # label columns and index
    nk_df.columns = ['n', 'k']
    nk_df.index.name = 'lambda'

    # create complex refractive index using interpolation form nkfile
    N = np.interp(lam, nk_df.index, nk_df['n'] + 1j*nk_df['k'])

    # if N.real or N.imag < 0, make it = 0
    N[N.real<0] = 0                + 1j*N[N.real<0].imag # real part = 0 (keep imag part)
    N[N.imag<0] = N[N.imag<0].real + 1j*0                # imag part = 0 (keep real part)
    
    # warning if extrapolated values
    if lam[ 0] < nk_df.index[0] :
        warn('Extrapolating from %.3f to %.3f' % (lam[0], nk_df.index[0]))
        
    if lam[-1] > nk_df.index[-1] :
        warn('Extrapolating from %.3f to %.3f' % (nk_df.index[-1], lam[-1]))
    
    # if lam was float (orginaly), convert N to a complex value
    return complex(N[0]) if lam_isfloat else N, nk_df
    # return N(lam), nk_df

def read_nk_yaml_from_ri_info(url):
    """
    Reads a YAML file containing 'nk' tabulated optical data from a URL and returns:
    - lam: ndarray of wavelengths
    - nk: ndarray of complex refractive indices (n + ik)
    """
    # Download YAML content
    response = requests.get(url)
    response.raise_for_status()

    # Parse YAML content
    yaml_data = yaml.safe_load(response.text)

    # Extract tabulated data block
    nk_text = yaml_data['DATA'][0]['data']

    # Read into DataFrame using regex-based separator
    nk_df = pd.read_csv(StringIO(nk_text), sep=r'\s+', names=['wavelength', 'n', 'k'])

    return nk_df


def get_ri_info(lam,shelf,book,page):
    '''
    Extract refractive index from refractiveindex.info database. This code
    uses the refidx package from Bejamin Vial (https://gitlab.com/benvial/refidx)

    Parameters
    ----------
    lam : ndarray
        Wavelengths to interpolate (um).
    shelf : string
        Name of the shelf (main, organic, glass, other, 3D)
    book : string
        Material name
    page: string
        Refractive index source   

    Returns
    -------
    N : ndarray
        Interpolated complex refractive index
    data: ndarray
        Original tabulated data from file
    '''

    url_root = 'https://refractiveindex.info/database/data/' 
    url = url_root  + shelf + '/'  + book  + '/nk/' + page + '.yml'
    nk_df = read_nk_yaml_from_ri_info(url)

    
    # Convert to NumPy arrays
    matLambda = nk_df['wavelength'].to_numpy()
    mat_nk = nk_df['n'].to_numpy() + 1j*nk_df['k'].to_numpy()

    # db = ri.DataBase()
    # mat = db.materials[shelf][book][page]
    # matLambda = np.array(mat.material_data["wavelengths"])
    # matN = np.array(mat.material_data["index"])
    N = np.interp(lam, matLambda, mat_nk)
    
    # warning if extrapolated values
    if lam[ 0] < matLambda[ 0] :
        warn('Extrapolating from %.3f to %.3f' % (lam[0], matLambda[ 0]))
        
    if lam[-1] > matLambda[-1] :
        warn('Extrapolating from %.3f to %.3f' % (matLambda[-1], lam[-1]))
    
    return N, nk_df
'''
    --------------------------------------------------------------------
                    dielectric constant models
    --------------------------------------------------------------------
'''
def eps_lorentz(A,gamma,E0,lam):
    '''
    Lorentz oscillator model for dielectric constant based on
    parameters from ellipsometry measurements.

    Parameters
    ----------
    A   : float
        Oscillator's amplitude  
    
    gamma  : float
        Broadening of the oscillator(eV)
    
    E0  : float
        Oscillator's resonant energy (eV)
        
    lam : ndarray
        Wavelengths range (um)

    Returns
    -------
    eps : ndarray (complex)
        Complex dielectric constant
    '''

    wp = np.sqrt(A*gamma*E0)
    return lorentz(0,wp,E0,gamma,lam)**2

def eps_gaussian(A,Br,E0,lam):
    '''
    Gaussian oscillator model for dielectric constant based on
    parameters from ellipsometry measurements. The model first calculates
    the imaginary part of epsilon, and the retrieves the real component
    using Krammers-Kronig model.

    Parameters
    ----------
    A   : float
        Absorption amplitude  
    
    Br  : float
        Broadening (eV)
    
    E0  : float
        Oscillator energy (eV)

    lam : ndarray
        Wavelengths range (um)

    Returns
    -------
    eps : ndarray (complex)
        Complex dielectric constant
    '''
    #  Gauss model as function of E (in eV)
    f = 0.5/np.sqrt(np.log(2)) # scaling constant
    eps_G = lambda E: A*np.exp(-(f*(E - E0)/Br)**2) \
                    - A*np.exp(-(f*(E + E0)/Br)**2)

    # get real and imaginary part of dielectric constant
    E = convert_units(lam,'um','eV')                               # lambda range in eV
    a, b = max(E0 - 5*Br,0), E0 + 5*Br                             # integration range
    eps_re = eps_real_kkr(lam,eps_G,int_range=(a, b), cshift=1e-4) # get real part from KK
    eps_im = eps_G(E)                                              # imaginary component

    return eps_re + 1j*eps_im

def eps_tauc_lorentz(A,C,E0,Eg,lam):
    '''
    Tauc-Lorentz oscillator model for dielectric constant based on
    parameters from ellipsometry measurements.

    Parameters
    ----------
    A   : float
        Oscillator's amplitude  
    
    C  : float
        Broadening of the oscillator(eV)
    
    E0  : float
        Oscillator's resonant energy (eV)

    Eg  : float
        Bandgap (eV)
        
    lam : ndarray
        Wavelengths range (um)

    Returns
    -------
    eps : ndarray (complex)
        Complex dielectric constant
    '''
    
    #  Tauc-Lorentz model as function of E (in eV)
    eps_TL = lambda E: 1/E*A*E0*C*(E - Eg)**2/ \
                 ((E**2 - E0**2)**2 + C**2*E**2)*(E > Eg)
    
    # get real and imaginary part of dielectric constant
    E = convert_units(lam,'um','eV')                                # lambda range in eV
    a, b = Eg-20*C, Eg + 20*C                                            # set integration range
    eps_re = eps_real_kkr(lam,eps_TL,int_range=(a, b), cshift=1e-3) # get real part from KK
    eps_im = eps_TL(E)                                              # imaginary component
    
    #------------------------------------------------------------------------------
    # # Alternative form (didn't work)
    #------------------------------------------------------------------------------
    # source: Luis V. Rodríguez-de Marcos and Juan I. Larruquert, 
    #           "Analytic optical-constant model derived from Tauc-Lorentz and Urbach tail," 
    #           Opt. Express 24, 28561-28572 (2016)
    # a = 0.09
    # b = E + 1j*a
    # d = np.sqrt(E0**2 - (C/2)**2) - 1j*C/2
    # F = lambda x,y,z: ((Eg + x)**2*np.log(Eg + x) - (Eg - x)**2*np.log(Eg - x))/ \
    #                     x*(x**2 - y**2)*(x**2 - z**2)

    # eps = 1 + A*E0*C/np.pi*(F(b,d,d.conjugate())) + F(d,d.conjugate(),b) + F(d.conjugate(),b,d)
    #------------------------------------------------------------------------------

    return eps_re + 1j*eps_im

def eps_ellipsometry(oscilator_file, lam):
    '''
    Computes dielectric constant using ellipsometry fitting parameters.
    
    Parameters
    ----------
    oscilator_file   : csv file
        File containing the ellisometry fitting parameters, sorted as:

            type | A | E0 (eV) | C (eV) | Eg (eV) | Br (eV) | gamma (eV)
        
        where:
            - type: oscillator model (Tauc-Lorentz, Gauss, Lorentz)
            - A: Amplitude of oscillator
            - E0: resonant energy (apply to all models)
            - C: Tauc-Lorentz broadening (0 otherwise)
            - Eg: Tauc-Lorentz Bandgap (0 otherwise)
            - Br: Gauss model broadening (0 otherwise)
            - gamma: Lorentz model decay (0 otherwise)
    
    lam  : ndarray or float
        Wavelength range (um)

    Returns
    -------
    eps : ndarray (complex)
        Complex dielectric constant
    '''
    
    ellip_data = pd.read_csv(oscilator_file)
    eps = complex(0,0)
    for idx, oscillator in ellip_data.iterrows():
        model, A, E0, C, Eg, Br, gamma = oscillator
        if model == 'Tauc-Lorentz':
            eps += eps_tauc_lorentz(A,C,E0,Eg,lam)
            
        if model == 'Gaussian':
            eps += eps_gaussian(A,Br,E0,lam)
            
        elif model == 'Lorentz':
            eps += eps_lorentz(A,gamma,E0,lam)
    
    return eps


def lorentz(epsinf,wp,wn,gamma,lam):
    '''
    Refractive index from Lorentz model

    Parameters
    ----------
    epsinf : float
        dielectric constant at infinity.
    wp : float
        Plasma frequency, in eV (wp^2 = Nq^2/eps0 m).
    wn : float
        Natural frequency in eV
    gamma : float
        Decay rate in eV
    lam : linear np.array
        wavelength spectrum in um

    Returns
    -------
    complex refractive index

    '''
    from .utils import convert_units
    w = convert_units(lam,'um','eV')  # conver from um to eV 
    
    return np.sqrt(epsinf + wp**2/(wn**2 - w**2 - 1j*gamma*w))


def drude(epsinf,wp,gamma,lam):
    '''
    Refractive index from Drude model

    Parameters
    ----------
    epsinf : float
        dielectric constant at infinity.
    wp : float
        Plasma frequency, in eV (wp^2 = Nq^2/eps0 m).
    gamma : float
        Decay rate in eV
    lam : linear np.array
        wavelength spectrum in um

    Returns
    -------
    complex refractive index

    '''
    # define constants
    eV = 1.602176634E-19          # eV to J (conversion)
    hbar = 1.0545718E-34          # J*s (plank's constan)
    
    
    w = 2*np.pi*3E14/lam*hbar/eV  # conver from um to eV 
    
    return np.sqrt(epsinf - wp**2/(w**2 + 1j*gamma*w))

def emt_brugg(fv_1,nk_1,nk_2):
    '''
    Effective permitivity based on Bruggersman theory
    
        Parameters
    ----------
    fv_1: float   
        filling fraction of material inclusions

    nk_1: ndarray
        refractive index of inclusions
    
    nk_2: ndarray
        refractive index of host

    Returns
    -------
    nk_eff: ndarray
        complex refractive index of effective media
    '''
    
    # check simple cases first
    if fv_1 == 0:     # no inclusions
        return nk_2
    elif fv_1 == 1:   # no host
        return nk_1

    # prepare variables
    fv_2 = 1 - fv_1
    eps_1, eps_2 = nk_1**2, nk_2**2 # convert refractive index to dielectric constants
    
    # check if eps_1 or eps_2 are scalar and convert both to 1D ndarray
    eps_1, eps_1_isscalar = _ndarray_check(eps_1)
    eps_2, eps_2_isscalar = _ndarray_check(eps_2)

    # eps_1 is scalar, create a constant array of len(eps_2)
    if   eps_1_isscalar and not eps_2_isscalar:
        eps_1 = eps_1*np.ones_like(eps_2)
        
    # eps_2 is scalar, create a constant array of len(eps_1)
    elif not eps_1_isscalar and eps_2_isscalar:
        eps_2 = eps_2*np.ones_like(eps_1)
    
    # both are ndarrays, assert they have same length
    else:
        assert len(eps_1) == len(eps_2), 'size of eps_1 and eps_2 must be equal'

    # compute effective dielectric constant ussing Bruggerman theory.
    eps_m = 1/4.*((3*fv_1 - 1)*eps_1 + (3*fv_2 - 1)*eps_2                           \
            - np.sqrt(((3*fv_1 - 1)*eps_1 + (3*fv_2 - 1)*eps_2)**2 + 8*eps_1*eps_2))
    
    for i in range(len(eps_m)):
        if eps_m[i].imag < 0  or (eps_m[i].imag < 1E-10 and eps_m[i].real < 0):
            eps_m[i] =  eps_m[i] + \
                1/2*np.sqrt(((3*fv_1 - 1)*eps_1[i] + (3*fv_2 - 1)*eps_2[i])**2 \
                + 8*eps_1[i]*eps_2[i]) 
    
    # if eps_1 and eps_2 were scalar, return a single scalar value
    if len(eps_m) == 1: return np.sqrt(eps_m[0])
    else :              return np.sqrt(eps_m)

def eps_real_kkr(lam, eps_imag, eps_inf = 0, int_range = (0, np.inf), cshift=1e-12):
    '''
    Computes real part of dielectric constant from its imaginary components 
    using Krammers-Kronig relation

    Parameters
    ----------
    lam: ndarray or float
         wavelength spectrum (in microns)
    
    eps_imag: ndarray, float or callable 
              imaginary component of refractive index (if ndarray, it must be same size as lam)
    
    eps_inf: float (default 0)
             dielectric constant at infinity
    
    int_range: 2D tupple (default 0, inf) 
               integration range (only for eps_inf is callable)
    
    cshift: float
            Small value to avoid singularity at integration

    Returns
    -------
    eps_real: ndarray or float
              real part of dielectric constant
    '''
    from .utils import convert_units

    lam, lam_isfloat = _ndarray_check(lam)
    cshift = complex(0, cshift)
    w_i = convert_units(lam,'um', 'eV')

    if  isinstance(eps_imag, Callable):
        a, b = int_range # set integration range
        def integration_element(w_r):
            factor = lambda w: w / (w**2 - w_r**2 + cshift)
            real_int = lambda w: (eps_imag(w) * factor(w)).real
            imag_int = lambda w: (eps_imag(w) * factor(w)).imag
            total = quad(real_int, a,b)[0] + 1j*quad(imag_int, a,b)[0]
            return eps_inf + (2/np.pi)*total
        
    elif isinstance(eps_imag, np.ndarray) or isinstance(eps_imag,float):
        eps_imag = _ndarray_check(eps_imag)[0]
        assert lam.shape == eps_imag.shape, 'input arrays must be same length'
    
        def integration_element(w_r):
            factor = - w_i / (w_i**2 - w_r**2 + cshift) # integration domains are swaped, so a "-"" sign is added
            total = np.trapz(eps_imag * factor, x=w_i)
            return eps_inf + (2/np.pi)*total
    else:
        raise TypeError('Unknown type for eps_imag')
    
    eps_real = np.real([integration_element(w_r) for w_r in w_i]).reshape(-1)
    
    if lam.shape == (1,):
        return float(eps_real[0])
    return float(eps_real[0]) if lam_isfloat else eps_real 
'''
    --------------------------------------------------------------------
                            Target functions
    --------------------------------------------------------------------
'''

#------------------------------------------------------------------------------
#                                   Inorganic
# refractive index of SiO2
SiO2 = lambda lam: get_nkfile(lam, 'sio2_Palik_Lemarchand2013', get_from_local_path = True)[0]

# refractive index of CaCO3
CaCO3 = lambda lam: get_nkfile(lam, 'CaCO3_Palik', get_from_local_path = True)[0]

# refractive index of BaF2
BaF2 = lambda lam: get_ri_info(lam, 'main', 'BaF2', 'Querry')[0]

# refractive index of TiO2
TiO2 = lambda lam: get_ri_info(lam,'main','TiO2','Siefke')[0]

# refractive index of ZnO
ZnO = lambda lam: get_ri_info(lam,'main','ZnO','Querry')[0]

# refractive index of MgO
MgO = lambda lam: get_nkfile(lam,'MgO_Palik', get_from_local_path = True)[0]

# refractive index of Alumina (AL2O3)
Al2O3 = lambda lam: get_ri_info(lam,'main','Al2O3','Querry-o')[0]

# refractive index of ZnS
ZnS = lambda lam: get_ri_info(lam,'main','ZnS','Querry')[0]

# refractive index of amorphous GeSbTe (GST)
GSTa = lambda lam: get_nkfile(lam, 'GSTa_Du2016', get_from_local_path = True)[0]

# refractive index of crystaline GeSbTe (GST)
GSTc = lambda lam: get_nkfile(lam, 'GSTc_Du2016', get_from_local_path = True)[0]

# refractive index of crystaline GeSbTe (GST)
GSTc = lambda lam: get_nkfile(lam, 'GSTc_Du2016', get_from_local_path = True)[0]


# refractive index of Monoclinic(cold) Vanadium Dioxide (VO2M)
# sputtered on SiO2 by default (film2)
VO2M = lambda lam, film = 2: get_nkfile(lam, 'VO2M_Wan2019(film%i)' % film, get_from_local_path = True)[0]

# refractive index of Rutile(hot) Vanadium Dioxide (VO2R)
# sputtered on SiO2 by default (film2)
VO2R = lambda lam, film = 2: get_nkfile(lam, 'VO2R_Wan2019(film%i)' % film, get_from_local_path = True)[0]

def VO2(lam,T, film=2 , Tphc = 73, WT = 3.1):
    '''
    Refractive index of temperatura dependent VO2.
    Reference: Wan, C. et al. Ann. Phys. 531, 1900188 (2019).

    Parameters
    ----------
    lam : ndarray
        Wavelength range (um).
    T : float
        Temperature of VO2 (°C).
    film : int, optional
        Film type according to reference (The default is 2):
         - film 1: Si+native oxide/VO2(70nm) (Sputtered). 
         - film 2: Si+native oxide/VO2(130nm) (Sputtered).
         - film 3: Saphire/VO2(120nm) (Sputtered). 
         - film 4: Si+native oxide/VO2(110nm) (Sol-gel). 
    Tphc : float, optional
        Transition temperature (°C). The default is 73.
    WT : float, optional
        Width of IMT phase change (ev). The default is 3.1.

    Returns
    -------
    Complex refractive index

    '''
    # set constants
    kB = 8.617333262E-5 # eV/K (Boltzmann constant)
    Tphc = Tphc + 273   # convert °C to K
    T = T + 273         # convert °C to K
    
    fv = 1/(1 + np.exp(WT/kB*(1/T - 1/Tphc)))
    eps_c = VO2M(lam,film)**2
    eps_h = VO2R(lam,film)**2
    
    eps = (1 - fv)*eps_c + fv*eps_h
    
    return np.sqrt(eps)

# refractive index of Silicon
Si   = lambda lam: get_nkfile(lam, 'si_Schinke2017', get_from_local_path = True)[0]

#------------------------------------------------------------------------------
#                                   Metals
# refractive index of Gold
gold = lambda lam: get_nkfile(lam, 'au_Olmon2012_evap', get_from_local_path = True)[0]

# refractive index of Silver
silver = lambda lam: get_nkfile(lam, 'ag_Ciesielski2017', get_from_local_path = True)[0]

# refractive index of Copper
Cu   = lambda lam: get_nkfile(lam, 'cu_Babar2015', get_from_local_path = True)[0]

# refractive index of Aluminium
Al   = lambda lam: get_nkfile(lam, 'al_Rakic1995', get_from_local_path = True)[0]

# refractive index of Magnesium
Mg   = lambda lam: get_ri_info(lam, 'main', 'Mg', 'Hagemann')[0]

#------------------------------------------------------------------------------
#                                   Polymers
# refractive index of HDPE
HDPE  = lambda lam: get_nkfile(lam, 'HDPE_Palik', get_from_local_path = True)[0]

# refractive index of HDPE
PDMS  = lambda lam: get_nkfile(lam, 'PDMS_Zhang2020_Querry1987', get_from_local_path = True)[0]

# refractive index of PMMA
PMMA = lambda lam: get_ri_info(lam,'organic','(C5H8O2)n - poly(methyl methacrylate)','Zhang-Tomson')[0]

# refractive index of PVDF-HFP
PVDF  = lambda lam: get_nkfile(lam, 'PVDF-HFP_Mandal2018', get_from_local_path = True)[0]

#------------------------------------------------------------------------------
#                                   Others
# refractive index of water
H2O  = lambda lam: get_nkfile(lam, 'h2o_Hale1973', get_from_local_path = True)[0]
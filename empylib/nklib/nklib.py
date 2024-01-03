# -*- coding: utf-8 -*-
"""
Library of tabulated refractive index

Created on Sun Nov  7 17:25:53 2021

@author: PanxoPanza
"""
import os
import platform
import numpy as np 
#from scipy.interpolate import interp1d
from warnings import warn

def get_nkfile(lam, MaterialName):
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
    # convert lambda to list
    if np.isscalar(lam): 
        lam = np.array([lam,])    
    
    # retrieve local path
    dir_separator = '\\' # default value
    if platform.system() == "Linux":    # linux
        dir_separator= '/'

    elif platform.system() == 'Darwin': # OS X
        dir_separator='/'

    elif platform.system() == "Windows":  # Windows...
        dir_separator='\\'

    dir_path = os.path.dirname(__file__) + dir_separator
    filename = dir_path + MaterialName + '.nk'
   
    # check if file exist
    assert os.path.isfile(filename), 'File not found'
    
    data = np.genfromtxt(filename)
    assert data.shape[1] <= 3, 'wrong file format'

    mat_data = {'wavelengths': data[:,0], 'index': data[:,1] + 1j*data[:,2]*(data[:,2] > 0)}

    # create complex refractive index using interpolation form nkfile
    N = np.interp(lam, mat_data['wavelengths'],mat_data['index'])
    
    # warning if extrapolated values
    if lam[ 0] < data[ 0,0] :
        warn('Extrapolating from %.3f to %.3f' % (lam[0], data[0,0]))
        
    if lam[-1] > data[-1,0] :
        warn('Extrapolating from %.3f to %.3f' % (data[-1,0], lam[-1]))
    
    return N, mat_data

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

    import refidx as ri
    db = ri.DataBase()
    mat = db.materials[shelf][book][page]
    matLambda = np.array(mat.material_data["wavelengths"])
    matN = np.array(mat.material_data["index"])
    N = np.interp(lam, matLambda, matN)
    
    # warning if extrapolated values
    if lam[ 0] < matLambda[ 0] :
        warn('Extrapolating from %.3f to %.3f' % (lam[0], matLambda[ 0]))
        
    if lam[-1] > matLambda[-1] :
        warn('Extrapolating from %.3f to %.3f' % (matLambda[-1], lam[-1]))
    
    return N, mat.material_data

'''
    --------------------------------------------------------------------
                    dielectric constant models
    --------------------------------------------------------------------
'''
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
    # define constants
    eV = 1.602176634E-19          # eV to J (conversion)
    hbar = 1.0545718E-34          # J*s (plank's constan)
    
    
    w = 2*np.pi*3E14/lam*hbar/eV  # conver from um to eV 
    
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

def emt_brugg(f1,nk_1,nk_2):
    '''
    Effective permitivity based on Bruggersman theory
    
        Parameters
    ----------
    nk_1: ndarray
        refractive index of inclussions
    
    nk_2: ndarray
        refractive index of host
    
    f1: float   
        filling fraction of material eps1

    Returns
    -------
    complex refractive index of effective media
    '''
    # check simple cases first
    if f1 == 0:     # no inclusions
        return nk_2
    elif f1 == 1:   # no host
        return nk_1

    # prepare variables
    f2 = 1 - f1
    eps_1, eps_2 = nk_1**2, nk_2**2 # convert refractive index to dielectric constants
    
    # if eps_1 and eps_2 are scalar convert both to 1D ndarray
    if np.isscalar(eps_1) and np.isscalar(eps_2):
        eps_1 = np.array([eps_1,])
        eps_2 = np.array([eps_2,])
    
    # else, scale the scalar variable with the ndarray. 
    else: 
        # eps_1 is scalar, create a constant array of len(eps_2)
        if   np.isscalar(eps_1):
            eps_1 = eps_1*np.ones(len(eps_2))
        
        # eps_2 is scalar, create a constant array of len(eps_1)
        elif np.isscalar(eps_2):
            eps_2 = eps_2*np.ones(len(eps_1))
        
        # if both are ndarrays, assert they are both equal length
        else:
            assert len(eps_1) == len(eps_2), 'size of eps_1 and eps_2 must be equal'

    # compute effective dielectric constant ussing Bruggerman theory.
    eps_m = 1/4.*((3*f1 - 1)*eps_1 + (3*f2 - 1)*eps_2                           \
            - np.sqrt(((3*f1 - 1)*eps_1 + (3*f2 - 1)*eps_2)**2 + 8*eps_1*eps_2))
    
    for i in range(len(eps_m)):
        if eps_m[i].imag < 0  or (eps_m[i].imag < 1E-10 and eps_m[i].real < 0):
            eps_m[i] =  eps_m[i] + \
                1/2*np.sqrt(((3*f1 - 1)*eps_1[i] + (3*f2 - 1)*eps_2[i])**2 \
                + 8*eps_1[i]*eps_2[i]) 
    
    # if eps_1 and eps_2 were scalar, return a single scalar value
    if len(eps_m) == 1: return np.sqrt(eps_m[0])
    else :              return np.sqrt(eps_m)
'''
    --------------------------------------------------------------------
                            Target functions
    --------------------------------------------------------------------
'''

#------------------------------------------------------------------------------
#                                   Oxides
# refractive index of SiO2
SiO2 = lambda lam: get_nkfile(lam, 'sio2_Palik_Lemarchand2013')[0]

# refractive index of TiO2
TiO2 = lambda lam: get_ri_info(lam,'main','TiO2','Siefke')[0]

# refractive index of ZnO
ZnO = lambda lam: get_ri_info(lam,'main','ZnO','Querry')[0]

# refractive index of Alumina (AL2O3)
Al2O3 = lambda lam: get_ri_info(lam,'main','Al2O3','Querry-o')[0]

# refractive index of ZnS
ZnS = lambda lam: get_ri_info(lam,'main','ZnS','Querry')[0]

# refractive index of amorphous GeSbTe (GST)
GSTa = lambda lam: get_nkfile(lam, 'GSTa_Du2016')[0]

# refractive index of crystaline GeSbTe (GST)
GSTc = lambda lam: get_nkfile(lam, 'GSTc_Du2016')[0]

# refractive index of crystaline GeSbTe (GST)
GSTc = lambda lam: get_nkfile(lam, 'GSTc_Du2016')[0]


# refractive index of Monoclinic(cold) Vanadium Dioxide (VO2M)
# sputtered on SiO2 by default (film2)
VO2M = lambda lam, film = 2: get_nkfile(lam, 'VO2M_Wan2019(film%i)' % film)[0]

# refractive index of Rutile(hot) Vanadium Dioxide (VO2R)
# sputtered on SiO2 by default (film2)
VO2R = lambda lam, film = 2: get_nkfile(lam, 'VO2R_Wan2019(film%i)' % film)[0]

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

#------------------------------------------------------------------------------
#                                   Inorganics
# refractive index of Silicon
Si   = lambda lam: get_nkfile(lam, 'si_Schinke2017')[0]

#------------------------------------------------------------------------------
#                                   Metals
# refractive index of Gold
gold = lambda lam: get_nkfile(lam, 'au_Olmon2012_evap')[0]

# refractive index of Silver
silver = lambda lam: get_nkfile(lam, 'ag_Ciesielski2017')[0]

# refractive index of Copper
Cu   = lambda lam: get_nkfile(lam, 'cu_Babar2015')[0]

# refractive index of Aluminium
Al   = lambda lam: get_nkfile(lam, 'al_Rakic1995')[0]

#------------------------------------------------------------------------------
#                                   Polymers
# refractive index of HDPE
HDPE  = lambda lam: get_nkfile(lam, 'HDPE_Palik')[0]

# refractive index of HDPE
PDMS  = lambda lam: get_nkfile(lam, 'PDMS_Zhang2020_Querry1987')[0]

# refractive index of PMMA
PMMA = lambda lam: get_ri_info(lam,'organic','(C5H8O2)n - poly(methyl methacrylate)','Zhang-Tomson')[0]

# refractive index of PVDF-HFP
PVDF  = lambda lam: get_nkfile(lam, 'PVDF-HFP_Mandal2018')[0]

#------------------------------------------------------------------------------
#                                   Others
# refractive index of water
H2O  = lambda lam: get_nkfile(lam, 'h2o_Hale1973')[0]
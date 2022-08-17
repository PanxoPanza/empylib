# -*- coding: utf-8 -*-
"""
Library of tabulated refractive index

Created on Sun Nov  7 17:25:53 2021

@author: PanxoPanza
"""
import os
import numpy as np 

def get_nkfile(lam, MaterialName):
    '''
    Reads an *.nk file and returns an interpolated
    1D numpy array with the complex refractive index
    of the material Hola chuquitin
    
    Parameters
    ----------
    lam : 1D numpy array
        Wavelengths to interpolate (um).
    MaterialName : string
        Name of *.nk file

    Returns
    -------
    1D numpy array
        Interpolated complex refractive index
    3D numpy array (optional)
        Original tabulated data from file
    '''
    
    # retrieve local path

    import platform

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

    # if lam is scalar length = 1, else get arrays length
    if np.isscalar(lam):
        len_lam = 1
    else:
        len_lam = len(lam)
    
    # create complex refractive index using interpolation form nkfile
    N = np.zeros((len_lam,2))
    for i in range(1,data.shape[1]):
        # interpolate the refractive index according to "lam"
        N[:,i-1] = np.interp(lam,data[:,0],data[:,i])
        
        # for extrapolated values make out = 0
        N[:,i-1] = N[:,i-1]*(lam<=np.max(data[:,0]))*(lam>=np.min(data[:,0])) 

    return N[:,0] + 1j*N[:,1] , data

'''
    --------------------------------------------------------------------
                            Target functions
    --------------------------------------------------------------------
'''

# refractive index of sio2
sio2 = lambda lam: get_nkfile(lam, 'sio2_Palik_Lemarchand2013')[0]

# refractive index of tio2
tio2 = lambda lam: get_nkfile(lam, 'tio2_Siefke2016')[0]

# refractive index of gold
au = lambda lam: get_nkfile(lam, 'au_Olmon2012_evap')[0]
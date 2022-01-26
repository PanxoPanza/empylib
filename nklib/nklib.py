# -*- coding: utf-8 -*-
"""
Library of tabulated refractive index

Created on Sun Nov  7 17:25:53 2021

@author: PanxoPanza
"""
import os
import numpy as np 

def read_nkfile(lam, MaterialName):
    '''
    Reads an *.nk file and returns an interpolated
    1D numpy array with the complex refractive index
    of the material
    
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
    '''
    # retrieve local path
    dir_path = os.path.dirname(__file__) + '\\'
    filename = dir_path + MaterialName + '.nk'
   
    # check if file exist
    assert os.path.isfile(filename), 'File not found'
    
    data = np.genfromtxt(filename)
    assert data.shape[1] < 3, 'wrong file format'
    
    N = np.zeros((len(lam),2))
    for i in range(1,data.shape[1]):
        # interpolate the refractive index according to "lam"
        N[:,i-1] = np.interp(lam,data[:,0],data[:,i])
        
        # for extrapolated values make out = 0
        N[:,i-1] = N[:,i-1]*(lam<=np.max(data[:,0]))*(lam>=np.min(data[:,0])) 

    return N[:,0] + 1j*N[:,1] , data

def sio2(lam, nkFileName = 'sio2_Kischkat') :
    '''
    Complex refractive index of SiO2

    Parameters
    ----------
    lam : 1D numpy array
        Wavelengths to interpolate (um).
    nkFileName : string, optional
        Name of *.nk file for SiO2. The default is 'sio2_Kischkat'.

    Returns
    -------
    1D numpy array
        Complex refractive index.

    '''
    return read_nkfile(lam,nkFileName)
    


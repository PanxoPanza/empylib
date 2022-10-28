# color_system.py
# -*- coding: utf-8 -*-
"""
This library contains the class ColorSystem that converts a spectra into RGB colors
Created on Wed 17 Aug, 2022

ref: https://scipython.com/blog/converting-a-spectrum-to-a-colour/

@author: PanxoPanza
"""

import os
import numpy as np
import platform

def xyz_from_xy(x, y):
    """Return the vector (x, y, 1-x-y)."""
    return np.array((x, y, 1-x-y))

class ColorSystem:
    """A class representing a color system.

    A color system defined by the CIE x, y and z=1-x-y coordinates of
    its three primary illuminants and its "white point".

    TODO: Implement gamma correction

    """
    # Configure directory path
    dir_separator = '\\' # default value
    if platform.system() == "Linux":        # linux
        dir_separator= '/'

    elif platform.system() == 'Darwin':     # OS X
        dir_separator='/'

    elif platform.system() == "Windows":    # Windows...
        dir_separator='\\'

    # The CIE color matching function for 380 - 780 nm in 5 nm intervals
    file_name = 'cie-cmf.txt'
    dir_path = os.path.dirname(__file__) + dir_separator
    lam_cmf = np.loadtxt(dir_path+file_name, usecols=(0)) # wavelength spectrum
    cmf = np.loadtxt(dir_path+file_name, usecols=(1,2,3)) # x, y, z  CIE colour matching functions

    def __init__(self, red, green, blue, white):
        """Initialise the colorSystem object.

        Pass vectors (ie NumPy arrays of shape (3,)) for each of the
        red, green, blue  chromaticities and the white illuminant
        defining the color system.

        """

        # Chromaticities
        self.red, self.green, self.blue = red, green, blue
        self.white = white
        
        # The chromaticity matrix (rgb -> xyz) and its inverse
        self.M = np.vstack((self.red, self.green, self.blue)).T 
        self.MI = np.linalg.inv(self.M)
        
        # White scaling array
        self.wscale = self.MI.dot(self.white)
        
        # xyz -> rgb transformation matrix
        self.T = self.MI / self.wscale[:, np.newaxis]
        
    def interp_internals(self,lam):
        '''
        Update internal cmf and lam spectra

        Parameters
        ----------
        lam : ndarray
            wavelength spectrum in microns.

        Returns
        -------
        None.

        '''
        lam = lam*1E3
        cmf_local = np.zeros((len(lam),3))
        cmf_local[:,0] = np.interp(lam,self.lam_cmf,self.cmf[:,0])
        cmf_local[:,1] = np.interp(lam,self.lam_cmf,self.cmf[:,1]) 
        cmf_local[:,2] = np.interp(lam,self.lam_cmf,self.cmf[:,2])
        self.lam_cmf = np.copy(lam)
        self.cmf = np.copy(cmf_local)
        

    def xyz_to_rgb(self, xyz, out_fmt=None):
        """Transform from xyz to rgb representation of color.

        The output rgb components are normalized on their maximum
        value. If xyz is out the rgb gamut, it is desaturated until it
        comes into gamut.

        By default, fractional rgb components are returned; if
        out_fmt='html', the HTML hex string '#rrggbb' is returned.

        """
        rgb = self.T.dot(xyz)
        if np.any(rgb < 0):
            # We're not in the RGB gamut: approximate by desaturating
            w = - np.min(rgb)
            rgb += w
        if not np.all(rgb==0):
            # Normalize the rgb vector
            rgb /= np.max(rgb)

        if out_fmt == 'html':
            return self.rgb_to_hex(rgb)
        return rgb

    def rgb_to_hex(self, rgb):
        """Convert from fractional rgb values to HTML-style hex string."""

        hex_rgb = (255 * rgb).astype(int)
        return '#{:02x}{:02x}{:02x}'.format(*hex_rgb)

    def spec_to_xyz(self, spec, lam=None):
        '''
        Convert a spectrum to an xyz point.

        Parameters
        ----------
        spec : ndarray
            spectral radianse.
            
        lam : ndarray, optional
            wavelength scpectrum in microns. The default is None. Size of lam
            must be equal to the size of spec

        Returns
        -------
        ndarray
            xyz color code.

        '''
        # set wavelength range
        if lam is None: 
            lam = self.lam_cmf
            cmf_local = np.copy(self.cmf)
        else: 
            lam = lam*1E3
            cmf_local = np.zeros((len(lam),3))
            cmf_local[:,0] = np.interp(lam,self.lam_cmf,self.cmf[:,0])
            cmf_local[:,1] = np.interp(lam,self.lam_cmf,self.cmf[:,1]) 
            cmf_local[:,2] = np.interp(lam,self.lam_cmf,self.cmf[:,2])
        
        assert len(lam) == len(spec), 'size of lam and spec must be equal'
        
        #XYZ = np.sum(spec[:, np.newaxis] * self.cmf, axis=0)
        XYZ = np.trapz(spec[:, np.newaxis] *cmf_local, x=lam, axis=0)
        den = np.sum(XYZ)
        if den == 0.:
            return XYZ
        return XYZ / den

    def spec_to_rgb(self, spec, lam = None, out_fmt=None):
        '''
        Convert a spectrum to an rgb value.
        
        Parameters
        ----------
        spec : ndarray
            spectral radiance, len(lam) == len(spec).
            
        lam : ndarray
             wavelength range in micrometers. The default is None. size of lam
             must be equal to the size of spec
             
        out_fmt : string, optional
            out_html='html' if conevting to hex format. The default is None.

        Returns
        -------
        ndarray
            color RGB code.

        '''
        
        xyz = self.spec_to_xyz(spec, lam)
        return self.xyz_to_rgb(xyz, out_fmt)

illuminant_D65 = xyz_from_xy(0.3127, 0.3291)
hdtv = ColorSystem(red=xyz_from_xy(0.67, 0.33),
                       green=xyz_from_xy(0.21, 0.71),
                       blue=xyz_from_xy(0.15, 0.06),
                       white=illuminant_D65)

smpte = ColorSystem(red=xyz_from_xy(0.63, 0.34),
                        green=xyz_from_xy(0.31, 0.595),
                        blue=xyz_from_xy(0.155, 0.070),
                        white=illuminant_D65)

srgb = ColorSystem(red=xyz_from_xy(0.64, 0.33),
                       green=xyz_from_xy(0.30, 0.60),
                       blue=xyz_from_xy(0.15, 0.06),
                       white=illuminant_D65)
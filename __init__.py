import os
from . import nklib
from . import ref_spectra
from . import miescattering
from . import waveoptics
from . import rad_transfer

__version__ = "0.1.2"
__author__ = 'Francisco Ramirez'
__credits__ = 'Universidad Adolfo Ibañez'

# standard constants
e_charge = 1.602176634E-19      # C (elementary charge)
hbar = 1.0545718E-34            # J*s (plank's constan)
speed_of_light = 299792458      # m/s (speed of light)
kBoltzmann = 1.38064852E-23     # J/K (Boltzman constant)

def convert_units(x, x_in, to):
    '''
    Convert units of a variable. Accepted units for conversion are:
        nanometers              : 'nm'
        micrometers             : 'um'
        recriprocal centimeters : 'cm^-1'
        frequency               : 'Hz'
        angular frequency       : 'rad/s'
        electron voltz          : 'eV'

    Parameters
    ----------
    x : ndarray
        list of values to convert.
    x_in : string
        units of the input variable.
    to : string
        conversion units.

    Returns
    -------
    ndarray
        coverted list of values.

    '''
    
    eV = 1.602176634E-19      # C (elementary charge)
    hbar = 1.0545718E-34      # J*s/rad (red. plank's constan)
    PI = 3.141592653589793
    c0 = speed_of_light       # m/s (speed of light)
    hbar = hbar/eV            # eV*s/rad (red. plank's constan)
    h = 2*PI*hbar             # eV/Hz (plank's constan)
    
    unit_dict = ['nm', 'um', 'cm^-1', 'Hz', 'rad/s', 'eV']
    
    assert x_in in unit_dict, 'Unkown unit: ' + x_in
    assert to in unit_dict, 'Unkown unit: ' + to
    
    unit_table = {
        ('nm','nm')       : x, 
        ('nm','um')       : x*1E-3, 
        ('nm','cm^-1')    : 1/x*1E7, 
        ('nm','Hz')       : c0/x*1E9,
        ('nm','rad/s')    : 2*PI*c0/x*1E9,
        ('nm','eV')       : h*c0/x*1E9, 
        
        ('um','nm')       : x*1E3, 
        ('um','um')       : x, 
        ('um','cm^-1')    : 1/x*1E4, 
        ('um','Hz')       : c0/x*1E6,
        ('um','rad/s')    : 2*PI*c0/x*1E6,
        ('um','eV')       : h*c0/x*1E6, 
        
        ('cm^-1','nm')    : 1/x*1E7, 
        ('cm^-1','um')    : 1/x*1E4,  
        ('cm^-1','cm^-1') : x, 
        ('cm^-1','Hz')    : x*c0*1E2,
        ('cm^-1','rad/s') : x*2*PI*c0*1E2,
        ('cm^-1','eV')    : x*h*c0*1E2, 
        
        ('Hz','nm')       : c0*1E9/x, 
        ('Hz','um')       : c0*1E6/x,
        ('Hz','cm^-1')    : x/(c0*1E2), 
        ('Hz','Hz')       : x,
        ('Hz','rad/s')    : 2*PI*x,
        ('Hz','eV')       : h*x, 
        
        ('rad/s','nm')    : 2*PI*c0/x*1E9, 
        ('rad/s','um')    : 2*PI*c0/x*1E6,
        ('rad/s','cm^-1') : x/(2*PI*c0*1E2), 
        ('rad/s','Hz')    : x/(2*PI),
        ('rad/s','rad/s') : x,
        ('rad/s','eV')    : hbar*x, 
        
        ('eV','nm')       : h*c0/x*1E9, 
        ('eV','um')       : h*c0/x*1E6,
        ('eV','cm^-1')    : x/(h*c0*1E2), 
        ('eV','Hz')       : x/h,
        ('eV','rad/s')    : x/hbar,
        ('eV','eV')       : x, 
        
        }
    
    return unit_table[(x_in, to)]
import numpy as np

# standard constants
e_charge = 1.602176634E-19      # C (elementary charge)
hbar = 1.0545718E-34            # J*s (plank's constan)
speed_of_light = 299792458      # m/s (speed of light)
kBoltzmann = 1.38064852E-23     # J/K (Boltzman constant)

def _ndarray_check(x):
    '''
    check if x is not ndarray. If so, convert x to a 1d ndarray
    '''
    
    if not isinstance(x, np.ndarray):
        return np.array([x]), True
    return x, False

# a function to convert units in electrodynamics
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

def _local_to_global_angles(theta_i, phi_i, beta_tilt, phi_tilt, restrict_to_upper_hemisphere=False):
    """
    Converts local spherical angles (theta_i, phi_i), defined relative to the normal of a tilted surface,
    into global spherical angles (theta, phi) defined with respect to the global vertical (z-axis).

    Parameters:
    -----------
    theta_i : float or ndarray
        Local zenith angle in radians (relative to the tilted surface normal).
    phi_i : float or ndarray
        Local azimuth angle in radians (relative to the tilted surface normal).
    beta : float
        Tilt angle of the surface with respect to the global vertical (radians).
    phi_tilt : float
        Azimuth direction of the surface tilt in the global frame (radians).
    restrict_to_upper_hemisphere : bool
        If True, any direction that ends up in the lower global hemisphere (theta > π/2) will be reflected upward.

    Returns:
    --------
    theta : float or ndarray
        Global zenith angle in radians (relative to vertical).
    phi : float or ndarray
        Global azimuth angle in radians (0 to 2π).
    """
    import numpy as np

    original_shape = np.shape(theta_i)
    theta_i = np.ravel(theta_i)
    phi_i = np.ravel(phi_i)

    # Tilted normal vector in global coordinates
    n_local = np.array([
        np.sin(beta_tilt) * np.cos(phi_tilt),
        np.sin(beta_tilt) * np.sin(phi_tilt),
        np.cos(beta_tilt)
    ])
    
    # Rotation axis: cross product between global z and tilted normal
    axis = np.cross([0, 0, 1], n_local)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-8:
        # No tilt: return inputs unchanged
        theta = theta_i.reshape(original_shape)
        phi = phi_i.reshape(original_shape)
        return theta.item() if theta.size == 1 else theta, phi.item() if phi.size == 1 else phi

    axis = axis / axis_norm  # normalize

    # Rodrigues' rotation matrix for rotation around arbitrary axis
    cos_b = np.cos(beta_tilt)
    sin_b = np.sin(beta_tilt)
    ux, uy, uz = axis

    R = np.array([
        [cos_b + ux**2 * (1 - cos_b),     ux*uy*(1 - cos_b) - uz*sin_b,  ux*uz*(1 - cos_b) + uy*sin_b],
        [uy*ux*(1 - cos_b) + uz*sin_b,    cos_b + uy**2 * (1 - cos_b),   uy*uz*(1 - cos_b) - ux*sin_b],
        [uz*ux*(1 - cos_b) - uy*sin_b,    uz*uy*(1 - cos_b) + ux*sin_b,  cos_b + uz**2 * (1 - cos_b)]
    ])

    # Convert local spherical to Cartesian coordinates
    x = np.sin(theta_i) * np.cos(phi_i)
    y = np.sin(theta_i) * np.sin(phi_i)
    z = np.cos(theta_i)
    v_local = np.stack([x, y, z], axis=0)  # shape: (3, N)

    # Rotate to global coordinates
    v_global = R @ v_local

    # Optional: reflect any downward-pointing vectors if restricting to upper hemisphere
    if restrict_to_upper_hemisphere:
        below = v_global[2] < 0
        v_global[:, below] *= -1

    # Convert to global spherical coordinates
    xg, yg, zg = v_global[0], v_global[1], v_global[2]
    theta = np.arccos(np.clip(zg, -1, 1))
    phi = np.mod(np.arctan2(yg, xg), 2 * np.pi)

    # Restore original shape
    theta = theta.reshape(original_shape)
    phi = phi.reshape(original_shape)

    if original_shape == ():
        return theta.item(), phi.item()
    return theta, phi
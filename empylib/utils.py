import numpy as np

# standard constants
e_charge = 1.602176634E-19      # C (elementary charge)
hbar = 1.0545718E-34            # J*s (plank's constan)
speed_of_light = 299792458      # m/s (speed of light)
kBoltzmann = 1.38064852E-23     # J/K (Boltzman constant)

def as_carray(x, name, nlam, val_type = complex):
        arr = np.asarray(x)
        if arr.ndim == 0:
            return np.full(nlam, val_type(arr), dtype=val_type)
        if arr.shape != (nlam,):
            raise ValueError(f"{name} must be scalar or have shape (len(lam),).")
        return arr.astype(val_type)

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

def _check_mie_inputs(lam=None, N_host=None, Np_shells=None, D=None):
    """
    Validate and normalize inputs for Mie / multilayer-sphere calculations.

    Parameters
    ----------
    lam : float or (nλ,) array-like of float, optional
        Wavelength(s) in micrometers (µm). If array-like, must be 1D with lam > 0.
        If omitted (None), spectral refractive indices cannot be used (only scalars allowed).
    N_host : complex or (nλ,) array-like of complex, optional
        Host refractive index. If array-like, length must equal len(lam).
        If scalar, it is broadcast to (nλ,). If lam is None and N_host is array-like, error.
    Np_shells : scalar complex, (nλ,) array-like of complex, list/tuple of those,
                or (n_layers, nλ) ndarray, optional
        Refractive index for each shell layer.
        Accepted forms:
          - scalar (single-layer, constant with λ)
          - 1D array (single-layer spectrum, length = len(lam))
          - list/tuple of scalars/1D arrays (one per layer; arrays must match len(lam))
          - 2D ndarray shaped (n_layers, nλ)
        If arrays are provided but lam is None, error.
    D : float or 1D array-like of float, optional
        OUTER diameter of each shell layer (µm).
        - scalar: single-layer (solid sphere)
        - 1D: multilayer shell outer diameters
        Must be strictly increasing for multilayer (inner < ... < outer) and all > 0.

    Returns
    -------
    lam_out : (nλ,) ndarray of float or None
        Wavelength grid (if provided), 1D and strictly positive.
    N_host_out : (nλ,) ndarray of complex or None
        Host refractive index aligned to lam_out (broadcast if scalar).
    Np_out : (n_layers, nλ) ndarray of complex or None
        Shell refractive indices stacked by layer (inner→outer). If D is provided,
        rows are sorted to match sorted(D).
    D_out : (n_layers,) ndarray of float or None
        Shell outer diameters (µm), sorted ascending.

    Notes
    -----
    - If both D and Np_shells are provided, their number of layers must match.
    - This function only checks shapes/values and performs broadcasting/sorting.
      It does not compute anything optical.
    """
    # ---- lam ----
    if lam is None:
        lam_out = None
    else:
        lam_arr = np.asarray(lam, dtype=float).ravel()
        if lam_arr.ndim != 1 or lam_arr.size == 0:
            raise ValueError("lam must be a 1D array (non-empty) or a scalar.")
        if not np.all(np.isfinite(lam_arr)) or np.any(lam_arr <= 0):
            raise ValueError("All wavelengths in lam must be finite and > 0 (µm).")
        lam_out = lam_arr

    nlam = None if lam_out is None else lam_out.size

    # ---- D (diameters) ----
    D_out = None
    if D is not None:
        if np.isscalar(D):
            D_list = [float(D)]
        else:
            try:
                D_list = [float(d) for d in np.asarray(D).ravel()]
            except Exception:
                raise TypeError("D must be a float or a 1D list/array of outer diameters (µm).")
            if len(D_list) == 0:
                raise ValueError("D cannot be an empty list/array.")
        if any((not np.isfinite(d)) or (d <= 0) for d in D_list):
            raise ValueError("All diameters in D must be finite and > 0 (µm).")
        if len(D_list) > 1 and any(D_list[i] >= D_list[i+1] for i in range(len(D_list)-1)):
            raise ValueError("For multilayer spheres, D must be strictly increasing (inner < ... < outer).")
        D_out = np.asarray(D_list, dtype=float)

    # ---- Np_shells (layers) ----
    Np_out = None
    if Np_shells is not None:
        # Normalize to a (n_layers, nλ) 2D array (or (n_layers, 1) if lam is None and scalars)
        # Accept scalar, 1D spectrum, list of those, or 2D array.
        def to_layer_array(x):
            # returns 1D array for a single layer (length nlam if available, else length 1)
            if x.ndim == 0:  # scalar
                if nlam is None:
                    return np.array([complex(x)], dtype=complex)
                return np.full(nlam, complex(x), dtype=complex)
            arr = np.asarray(x, dtype=complex).ravel()
            if lam_out is None:
                raise ValueError("Spectral Np_shells provided but lam is None. Provide lam.")
            if arr.size != nlam:
                raise ValueError(f"A spectral layer has length {arr.size}, expected len(lam)={nlam}.")
            return arr

        if isinstance(Np_shells, list):
            if len(Np_shells) == 0:
                raise ValueError("Np_shells list cannot be empty.")
            layers = [to_layer_array(x) for x in Np_shells]
            # All layers must have same spectral length (nlam) if lam given
            L = [arr.size for arr in layers]
            if nlam is not None and any(Li != nlam for Li in L):
                raise ValueError("All spectral layers in Np_shells must have length len(lam).")
            Np_out = np.vstack([arr.reshape(1, -1) for arr in layers]).astype(complex)
        else:
            arr = np.asarray(Np_shells)
            if arr.ndim == 0:  # scalar
                Np_out = to_layer_array(arr).reshape(1, -1)
            elif arr.ndim == 1:  # single-layer spectrum
                Np_out = to_layer_array(arr).reshape(1, -1)
            elif arr.ndim == 2:
                # Expect shape (n_layers, nlam)
                if lam_out is None:
                    raise ValueError("2D Np_shells provided but lam is None. Provide lam.")
                if arr.shape[1] != nlam:
                    raise ValueError(f"Np_shells second dimension must equal len(lam)={nlam}.")
                if not np.all(np.isfinite(arr)):
                    raise ValueError("Np_shells contains non-finite values.")
                Np_out = arr.astype(complex)
            else:
                raise TypeError("Np_shells must be scalar, 1D array, list/tuple of scalars/1D arrays, or 2D array.")

    # ---- Cross-check layers vs diameters ----
    if (Np_out is not None) and (D_out is not None):
        n_layers_n = Np_out.shape[0]
        n_layers_d = D_out.size
        if n_layers_n != n_layers_d:
            raise ValueError(
                f"Number of layers mismatch: len(D)={n_layers_d} but Np_shells has {n_layers_n} layer(s)."
            )
        # Sort by D ascending and reorder Np_out rows accordingly
        order = np.argsort(D_out)
        D_out = D_out[order]
        Np_out = Np_out[order, :]

    # ---- N_host ----
    N_host_out = None
    if N_host is not None:
        if np.isscalar(N_host):
            if nlam is None:
                N_host_out = np.array([complex(N_host)], dtype=complex)
            else:
                N_host_out = np.full(nlam, complex(N_host), dtype=complex)
        else:
            if lam_out is None:
                raise ValueError("Spectral N_host provided but lam is None. Provide lam.")
            arr = np.asarray(N_host, dtype=complex).ravel()
            if arr.size != nlam:
                raise ValueError(f"N_host length must equal len(lam)={nlam}.")
            N_host_out = arr

    # ---- Final NaN/Inf guards ----
    for name, arr in [("lam", lam_out), ("N_host", N_host_out), ("Np_shells", Np_out), ("D", D_out)]:
        if arr is None:
            continue
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values.")

    return lam_out, N_host_out, Np_out, D_out

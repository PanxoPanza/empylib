import numpy as _np
from typing import Union as _Union, Optional as _Optional
from inspect import Signature as _Signature

# standard constants
e_charge = 1.602176634E-19      # C (elementary charge)
hbar = 1.0545718E-34            # J*s (plank's constan)
speed_of_light = 299792458      # m/s (speed of light)
kBoltzmann = 1.38064852E-23     # J/K (Boltzman constant)

def _as_carray(x, name, nlam, val_type = complex):
        arr = _np.asarray(x)
        if arr.ndim == 0:
            return _np.full(nlam, val_type(arr), dtype=val_type)
        if arr.shape != (nlam,):
            raise ValueError(f"{name} must be scalar or have shape (len(lam),).")
        return arr.astype(val_type)

def _ndarray_check(x):
    '''
    check if x is not ndarray. If so, convert x to a 1d ndarray
    '''
    
    if not isinstance(x, _np.ndarray):
        return _np.array([x]), True
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

    original_shape = _np.shape(theta_i)
    theta_i = _np.ravel(theta_i)
    phi_i = _np.ravel(phi_i)

    # Tilted normal vector in global coordinates
    n_local = _np.array([
        _np.sin(beta_tilt) * _np.cos(phi_tilt),
        _np.sin(beta_tilt) * _np.sin(phi_tilt),
        _np.cos(beta_tilt)
    ])
    
    # Rotation axis: cross product between global z and tilted normal
    axis = _np.cross([0, 0, 1], n_local)
    axis_norm = _np.linalg.norm(axis)

    if axis_norm < 1e-8:
        # No tilt: return inputs unchanged
        theta = theta_i.reshape(original_shape)
        phi = phi_i.reshape(original_shape)
        return theta.item() if theta.size == 1 else theta, phi.item() if phi.size == 1 else phi

    axis = axis / axis_norm  # normalize

    # Rodrigues' rotation matrix for rotation around arbitrary axis
    cos_b = _np.cos(beta_tilt)
    sin_b = _np.sin(beta_tilt)
    ux, uy, uz = axis

    R = _np.array([
        [cos_b + ux**2 * (1 - cos_b),     ux*uy*(1 - cos_b) - uz*sin_b,  ux*uz*(1 - cos_b) + uy*sin_b],
        [uy*ux*(1 - cos_b) + uz*sin_b,    cos_b + uy**2 * (1 - cos_b),   uy*uz*(1 - cos_b) - ux*sin_b],
        [uz*ux*(1 - cos_b) - uy*sin_b,    uz*uy*(1 - cos_b) + ux*sin_b,  cos_b + uz**2 * (1 - cos_b)]
    ])

    # Convert local spherical to Cartesian coordinates
    x = _np.sin(theta_i) * _np.cos(phi_i)
    y = _np.sin(theta_i) * _np.sin(phi_i)
    z = _np.cos(theta_i)
    v_local = _np.stack([x, y, z], axis=0)  # shape: (3, N)

    # Rotate to global coordinates
    v_global = R @ v_local

    # Optional: reflect any downward-pointing vectors if restricting to upper hemisphere
    if restrict_to_upper_hemisphere:
        below = v_global[2] < 0
        v_global[:, below] *= -1

    # Convert to global spherical coordinates
    xg, yg, zg = v_global[0], v_global[1], v_global[2]
    theta = _np.arccos(_np.clip(zg, -1, 1))
    phi = _np.mod(_np.arctan2(yg, xg), 2 * _np.pi)

    # Restore original shape
    theta = theta.reshape(original_shape)
    phi = phi.reshape(original_shape)

    if original_shape == ():
        return theta.item(), phi.item()
    return theta, phi

def _check_mie_inputs(lam=None, N_host=None, Np_shells=None, D=None, *, size_dist=None):
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
    D : float, 1D array-like of float, or list of those, optional
        Outer diameter(s) per layer (µm). Semantics:
          - float: single-layer, monodisperse
          - 1D array: single-layer, polydisperse (size distribution)
          - list of floats: multilayer, monodisperse
          - list of 1D arrays: multilayer, polydisperse (all arrays must be same length)
        For multilayer monodisperse (list of floats): strictly increasing.
        For multilayer polydisperse (list of arrays): element-wise strictly increasing across layers.
    size_dist : None or 1D array-like of float, optional (default None)
        Size-distribution weights. **If None, the distribution must be monodisperse** (i.e., all layers
        length==1). For polydisperse D, you must provide size_dist with matching length; it will be normalized.

    Returns
    -------
    lam_out : (nλ,) ndarray of float or None
    N_host_out : (nλ,) ndarray of complex or None
    Np_out : (n_layers, nλ) ndarray of complex or None
        Layers ordered inner→outer. If D is provided, rows are reordered accordingly.
    D_out : list[ndarray]
        One ndarray per layer. For monodisperse layers, each entry is a length-1 array.
    size_dist_out : (n_bins,) ndarray of float
        Normalized size-distribution vector. For monodisperse, array([1.0]).

    Notes
    -----
    - If both D and Np_shells are provided, their number of layers must match.
    - If size_dist is None, inputs must be monodisperse (no arrays of length > 1 in D).
    """
    # ---- lam ----
    if lam is None:
        lam_out = None
    else:
        lam_arr = _np.asarray(lam, dtype=float).ravel()
        if lam_arr.ndim != 1 or lam_arr.size == 0:
            raise ValueError("lam must be a 1D array (non-empty) or a scalar.")
        if not _np.all(_np.isfinite(lam_arr)) or _np.any(lam_arr <= 0):
            raise ValueError("All wavelengths in lam must be finite and > 0 (µm).")
        lam_out = lam_arr

    nlam = None if lam_out is None else lam_out.size

    # ---- D (diameters) → list[ndarray]
    D_out = None
    n_bins = None  # number of size bins if polydisperse

    if D is not None:
        def _as_1d_array_positive(x):
            arr = _np.asarray(x, dtype=float).ravel()
            if arr.size == 0:
                raise ValueError("D arrays must be non-empty.")
            if not _np.all(_np.isfinite(arr)) or _np.any(arr <= 0):
                raise ValueError("All diameters in D must be finite and > 0 (µm).")
            return arr

        if _np.isscalar(D) or (isinstance(D, _np.ndarray) and _np.asarray(D).ndim == 0):
            # single-layer monodisperse
            D_out = [_np.array([float(D)], dtype=float)]

        elif isinstance(D, (list, tuple)):
            if len(D) == 0:
                raise ValueError("D cannot be an empty list.")
            D_list = []
            for d in D:
                if _np.isscalar(d) or (isinstance(d, _np.ndarray) and _np.asarray(d).ndim == 0):
                    D_list.append(_np.array([float(d)], dtype=float))
                else:
                    arr = _as_1d_array_positive(d)
                    D_list.append(arr)

            lengths = _np.array([a.size for a in D_list], dtype=int)
            if _np.any(lengths > 1):
                # polydisperse multilayer
                n_bins = int(lengths.max())
                if not _np.all(lengths == n_bins):
                    raise ValueError("For multilayer polydisperse, all layer arrays in D must have the same length.")
                D_stack = _np.vstack(D_list)
                if _np.any(_np.diff(D_stack, axis=0) <= 0):
                    raise ValueError("For multilayer polydisperse, diameters must be element-wise strictly increasing across layers.")
            else:
                # multilayer monodisperse: strictly increasing floats
                mono_vals = _np.array([a.item() for a in D_list], dtype=float)
                if _np.any(~_np.isfinite(mono_vals)) or _np.any(mono_vals <= 0):
                    raise ValueError("All diameters in D must be finite and > 0 (µm).")
                if _np.any(_np.diff(mono_vals) <= 0):
                    raise ValueError("For multilayer monodisperse, D must be strictly increasing (inner < ... < outer).")

            D_out = D_list

        else:
            # single-layer polydisperse (1D array)
            arr = _as_1d_array_positive(D)
            D_out = [arr]

        if n_bins is None:
            n_bins = int(max(a.size for a in D_out))

        # Enforce either all mono or all share n_bins>1
        if not (all(a.size == 1 for a in D_out) or all(a.size == n_bins for a in D_out)):
            raise ValueError("All layers must be monodisperse (length=1) or all polydisperse with a common length.")

    # ---- Np_shells (layers) → (n_layers, nλ)
    Np_out = None
    if Np_shells is not None:
        def to_layer_array(x):
            xa = _np.asarray(x)
            if xa.ndim == 0:  # scalar
                if nlam is None:
                    return _np.array([complex(xa)], dtype=complex)
                return _np.full(nlam, complex(xa), dtype=complex)
            arr = _np.asarray(xa, dtype=complex).ravel()
            if lam_out is None:
                raise ValueError("Spectral Np_shells provided but lam is None. Provide lam.")
            if arr.size != nlam:
                raise ValueError(f"A spectral layer has length {arr.size}, expected len(lam)={nlam}.")
            return arr

        if isinstance(Np_shells, (list, tuple)):
            if len(Np_shells) == 0:
                raise ValueError("Np_shells list cannot be empty.")
            layers = [to_layer_array(x) for x in Np_shells]
            if nlam is not None and any(arr.size != nlam for arr in layers):
                raise ValueError("All spectral layers in Np_shells must have length len(lam).")
            Np_out = _np.vstack([arr.reshape(1, -1) for arr in layers]).astype(complex)
        else:
            arr = _np.asarray(Np_shells)
            if arr.ndim == 0:
                Np_out = to_layer_array(arr).reshape(1, -1)
            elif arr.ndim == 1:
                Np_out = to_layer_array(arr).reshape(1, -1)
            elif arr.ndim == 2:
                if lam_out is None:
                    raise ValueError("2D Np_shells provided but lam is None. Provide lam.")
                if arr.shape[1] != nlam:
                    raise ValueError(f"Np_shells second dimension must equal len(lam)={nlam}.")
                if not _np.all(_np.isfinite(arr)):
                    raise ValueError("Np_shells contains non-finite values.")
                Np_out = arr.astype(complex)
            else:
                raise TypeError("Np_shells must be scalar, 1D array, list/tuple of scalars/1D arrays, or 2D array.")

    # ---- Cross-check layers vs diameters ----
    if (Np_out is not None) and (D_out is not None):
        if Np_out.shape[0] != len(D_out):
            raise ValueError(
                f"Number of layers mismatch: len(D)={len(D_out)} but Np_shells has {Np_out.shape[0]} layer(s)."
            )

    # ---- N_host ----
    N_host_out = None
    if N_host is not None:
        if _np.isscalar(N_host):
            if nlam is None:
                N_host_out = _np.array([complex(N_host)], dtype=complex)
            else:
                N_host_out = _np.full(nlam, complex(N_host), dtype=complex)
        else:
            if lam_out is None:
                raise ValueError("Spectral N_host provided but lam is None. Provide lam.")
            arr = _np.asarray(N_host, dtype=complex).ravel()
            if arr.size != nlam:
                raise ValueError(f"N_host length must equal len(lam)={nlam}.")
            N_host_out = arr

    # ---- size_dist (normalize & validate) ----
    # If size_dist is None, inputs must be monodisperse.
    if D_out is None:
        # Without D, default to None
        size_dist_out = None
    else:
        is_mono = all(a.size == 1 for a in D_out)
        if size_dist is None:
            if not is_mono:
                raise ValueError(
                    "size_dist is None but D is polydisperse. "
                    "Provide size_dist with length equal to the number of size bins."
                )
            size_dist_out = None
        else:
            sd = _np.asarray(size_dist, dtype=float).ravel()
            if not _np.all(_np.isfinite(sd)) or _np.any(sd < 0) or sd.size == 0:
                raise ValueError("size_dist must contain non-negative finite values.")
            if is_mono:
                if sd.size != 1:
                    raise ValueError("For monodisperse D, size_dist must be length 1.")
                s = sd.sum()
                if s <= 0:
                    raise ValueError("size_dist sum must be > 0.")
                size_dist_out = sd / s
            else:
                n_bins = D_out[0].size  # all layers share same length if polydisperse
                if sd.size != n_bins:
                    raise ValueError(f"size_dist length {sd.size} must match number of size bins {n_bins}.")
                s = sd.sum()
                if s <= 0:
                    raise ValueError("size_dist sum must be > 0.")
                size_dist_out = sd / s

    # ---- Final NaN/Inf guards ----
    for name, arr in [("lam", lam_out), ("N_host", N_host_out), ("Np_shells", Np_out)]:
        if arr is None:
            continue
        if not _np.all(_np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values.")

    if D_out is not None:
        for a in D_out:
            if not _np.all(_np.isfinite(a)):
                raise ValueError("D contains non-finite values.")

    return lam_out, N_host_out, Np_out, D_out, size_dist_out

def _check_theta(theta: _Optional[_Union[float, _np.ndarray]],
                 n_theta: int = 181) -> _np.ndarray:
    """
    Validate and format the scattering angle array (theta).

    Parameters
    ----------
    theta : float or ndarray or None
        Scattering angle(s) in radians. If None, generates a dense grid from 0 to π.
    n_theta : int, optional
        Number of points in the angular grid if `theta` is None.

    Returns
    -------
    theta : _np.ndarray
        1D array of angles in radians, shape (n_theta,).
    """
    if theta is None:
        theta = _np.linspace(0.0, 2*_np.pi, max(int(n_theta), 5))
    
    elif _np.isscalar(theta):
        theta = _np.array([float(theta)])
    
    else:
        theta = _np.asarray(theta, dtype=float).ravel()

    return theta

# decorator to hide function signature
def _hide_signature(func):
    try:
        func.__signature__ = _Signature()
    except Exception:
        pass
    return func

def _warn_extrapolation(lam_arr, lo, hi, label="", quantity=""):
    lam_min = float(_np.min(lam_arr))
    lam_max = float(_np.max(lam_arr))
    if lam_min < lo and lam_max > hi:
        print(
            f"Extrapolating {label} {quantity} (requested {lam_min:.3f}–{lam_max:.3f} µm; "
            f"data {lo:.3f}–{hi:.3f} µm)"
            )
        
    else:
        if lam_min < lo:
            print(
                f"Extrapolating {label} {quantity} below tabulated range "
                f"(requested min {lam_min:.3f} µm; data starts {lo:.3f} µm)",
               )
        if lam_max > hi:
            print(
                f"Extrapolating {label} {quantity} above tabulated range "
                f"(requested max {lam_max:.3f} µm; data ends {hi:.3f} µm)",
            )
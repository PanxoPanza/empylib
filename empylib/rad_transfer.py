# -*- coding: utf-8 -*-
"""
Library of radiative transfer function

Created on Sun Nov  7 17:25:53 2021

@author: PanxoPanza
"""
import os
import sys

# empylib_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# sys.path.insert(0,empylib_folder)

import numpy as _np
from . import miescattering as mie
from . import waveoptics as wv
import iadpython as _iad
import pandas as pd
from typing import Union as _Union, Optional as _Optional
from .utils import as_carray

def T_beer_lambert(lam,theta, tfilm, Nlayer,fv,D,Np):
    '''
    Transmittance and reflectance from Beer-Lamberts law for a film with 
    spherical particles. Reflectance is cokmputed from classical formulas for 
    incoherent light incident on a slab between two semi-infinite media 
    (no scattering is considered for this parameter)

    Parameters
    ----------
    lam : ndaray
        Wavelength range in microns (um).
        
    theta : float
        Angle of incidence in radians (rad).
        
    tfilm : float
        Film Thickness in milimiters (mm).
        
    Nlayer : tuple
        Refractive index above, in, and below the film. Length of 
        N must be 3.
        
    fv : float
        Particle's volume fraction.
        
    D : float
        Particle's diameter in microns (um)
    
    Np : ndarray or float
        Refractive index of particles. If ndarray, the size must be equal to
        len(lam)

    Returns
    -------    
    Ttot : ndarray
        Spectral total transmisivity
        
    Rtot : ndarray
        Spectral total reflectivity
        
    Tspec : ndarray
        Spectral specular transmisivity

    '''
    if _np.isscalar(lam): lam = _np.array([lam]) # convert lam to ndarray

    assert isinstance(Nlayer, tuple), 'Nlayers must be on tuple format of dim = 3'
    assert len(Nlayer) == 3, 'length of Nlayer must be == 3'
    if not _np.isscalar(Np):
        assert len(Np) == len(lam), 'Np must be either scalar or an ndarray of size len(lam)'
    
    # convert all refractive index to arrays of size len(lam)
    # store result into a list
    N = []
    for Ni in Nlayer:
        if _np.isscalar(Ni): 
            Ni = _np.ones(len(lam))*Ni
        else: 
            assert len(Ni) == len(lam), 'refractive index must be either scalar or ndarray of size len(lam)'
        N.append(Ni)
    
    tfilm = tfilm*1E3 # convert mm to micron units
    
    Rp, Tp = wv.incoh_multilayer(lam, theta, N, tfilm, pol = 'TM')
    Rs, Ts = wv.incoh_multilayer(lam, theta, N, tfilm, pol = 'TE')
    T    = (Ts + Tp)/2
    Rtot = (Rp + Rs)/2
    
    # Get extinction and scattering efficiency of the sphere
    qext, qsca = mie.scatter_efficiency(lam, N[1], Np, D)[:2]
    qabs = qext - qsca # absorption efficiency
    
    Ac = _np.pi*D**2/4 # cross section area of sphere
    Vp = _np.pi*D**3/6 # volume of sphere
    cabs = Ac*qabs # absorption cross section
    cext = Ac*qext # extinction cross section
    
    theta1 = wv.snell(N[0],N[1], theta)
    # theta1 = _np.zeros(len(lam),dtype=complex)
    # for i in range(len(lam)):
    #     theta1[i] = wv.snell(N[0][i],N[1][i], theta)
        
    Ttot = T*_np.exp(-fv/Vp*cabs*tfilm/_np.cos(theta1.real))
    Tspec = T*_np.exp(-fv/Vp*cext*tfilm/_np.cos(theta1.real))

    return Ttot, Rtot, Tspec

def adm_sphere(
    lam: _Union[float, _np.ndarray],
    tfilm: float,
    fv: float,
    D: float,
    Np: _Union[float, _np.ndarray],
    Nh: _Union[float, _np.ndarray],
    *,
    Nup: _Union[float, _np.ndarray] = 1.0,
    Ndw: _Union[float, _np.ndarray] = 1.0,
    diffuse: bool = False,
    effective_medium: bool = True  # whether to compute effective Nh via Bruggeman
    ):
    """
    Compute spectral reflectance/transmittance of a film containing non-interacting **spherical particles**
    using the adding–doubling method (IAD).

    This wraps your single-particle Mie solver to obtain scattering/absorption **coefficients**
    (μ_s, μ_a) from efficiencies, then calls `adm(...)` to propagate through a slab with
    multiple scattering.

    Parameters
    ----------
    lam : array-like of float, shape (nλ,)
        Wavelengths in micrometers (µm). Must be > 0.
    tfilm : float
        Film thickness in millimeters (mm). Must be ≥ 0.
    fv : float
        Particle **volume fraction** (dimensionless). Must be in [0, 1).
    D : float or list[float]
        Particle outer diameter in micrometers (µm). If a list is provided, it is
        interpreted as a **multilayer sphere** with strictly increasing shell outer
        diameters; the last value is the overall (outermost) diameter.
    Np : complex or array-like of complex, shape (nλ,)
        Particle refractive index (can be complex, wavelength-dependent). If scalar,
        it is broadcast over `lam`.
    Nh : complex or array-like of complex, shape (nλ,)
        Host (film matrix) refractive index (can be complex). If scalar, broadcast.
    Nup : complex or array-like of complex, optional (default 1.0)
        Refractive index **above** the film (ambient). Scalar or shape (nλ,).
    Ndw : complex or array-like of complex, optional (default 1.0)
        Refractive index **below** the film (substrate). Scalar or shape (nλ,).
    diffuse : bool, optional (default False)
        If True, forces the asymmetry parameter g → 0 (diffuse approximation)
        when calling IAD.

    Returns
    -------
    Ttot : ndarray, shape (nλ,)
        Total (hemispherical) transmittance at normal incidence.
    Rtot : ndarray, shape (nλ,)
        Total (hemispherical) reflectance at normal incidence.
    Tspec : ndarray, shape (nλ,)
        Specular (unscattered) transmittance.
    Rspec : ndarray, shape (nλ,)
        Specular (unscattered) reflectance.

    Notes
    -----
    - Units: `k_sca, k_abs` are internally converted from µm⁻¹ to mm⁻¹ for IAD.
    - Efficiencies `Q` are turned into cross sections via the geometric area of the
      **outer diameter** (for multilayer spheres).
    - If `qabs = qext - qsca` yields small negative values (numerics), they are clipped to 0.
    - This function assumes **independent scattering** when forming μ from single-particle Mie.
    """
    # ---------- shape & value checks ----------
    if not _np.isscalar(tfilm):
        raise ValueError("tfilm must be a scalar thickness in mm.")
    if tfilm < 0:
        raise ValueError("tfilm must be ≥ 0 (mm).")

    if not (0 <= float(fv) < 1):
        raise ValueError("fv (volume fraction) must be in [0, 1).")

    nlam = lam.size

    Np_arr  = as_carray(Np,  "Np", nlam, val_type=complex)
    Nh_arr  = as_carray(Nh,  "Nh", nlam, val_type=complex)
    Nup_arr = as_carray(Nup, "Nup", nlam, val_type=complex)
    Ndw_arr = as_carray(Ndw, "Ndw", nlam, val_type=complex)

    if effective_medium:
        Nh_arr = _nk.emt_brugg(fv, Np_arr, Nh_arr)

    # ---------- Mie: efficiencies and g ----------
    qext, qsca, gcos = mie.scatter_efficiency(lam, Nh_arr, Np_arr, D)

    qext = _np.asarray(qext, dtype=float)
    qsca = _np.asarray(qsca, dtype=float)
    gcos = _np.asarray(gcos, dtype=float)

    if qext.shape != (nlam,) or qsca.shape != (nlam,) or gcos.shape != (nlam,):
        raise ValueError("mie.scatter_efficiency must return 1D arrays matching lam.")

    # absorption efficiency (clip small negatives)
    qabs = qext - qsca
    qabs[qabs < 0] = 0.0

    # ---------- particle geometry, cross sections ----------
    # Geometric area and volume based on OUTER diameter
    D_out = D[-1]
    Ac = _np.pi * (D_out / 2.0) ** 2        # [µm²]
    Vp = (_np.pi / 6.0) * (D_out ** 3)      # [µm³]

    # cross sections per particle [µm²]
    Csca = qsca * Ac
    Cabs = qabs * Ac

    # coefficients [µm⁻¹] using Case A (number-fraction implicit; here monodisperse)
    k_sca = (fv / Vp) * Csca
    k_abs = (fv / Vp) * Cabs

    # diffuse approx (optional)
    if diffuse:
        gcos = _np.zeros_like(gcos)

    # ---------- call IAD wrapper ----------
    Ttot, Rtot, Tspec, Rspec = adm(lam, tfilm, k_sca, k_abs, Nh_arr, 
                                   gcos=gcos, Nup=Nup_arr, Ndw=Ndw_arr)

    return Ttot, Rtot, Tspec, Rspec

def adm_poly_sphere(lam: _Union[float, _np.ndarray], # wavelengths [µm]
                    tfilm: float, fv: float, diameters: _Union[float, _np.ndarray], # sphere diameters [µm] 
                    size_dist: _Union[float, _np.ndarray], # number-fraction weights p_i
                    Np: _Union[float, _np.ndarray],  # particle refractive index
                    Nh: _Union[float, _np.ndarray], # host refractive index
                    *, 
                    Nup: _Union[float, _np.ndarray] = 1.0, # refractive index above
                    Ndw: _Union[float, _np.ndarray] = 1.0, # refractive index below
                    effective_medium: bool = True  # whether to compute effective Nh via Bruggeman
                    ):
    """
    Radiative transfer (IAD) for a film with a **polydisperse** ensemble of hard spheres.

    Pipeline:
      1) Use `mie.poly_sphere_cross_section` to get size-averaged cross sections per particle
         (⟨C_sca⟩, ⟨C_abs⟩) and the ensemble phase function vs angle.
      2) Convert to **coefficients** μ_s, μ_a via number density n_tot = f_v / ⟨V⟩.
      3) Call `adm(...)` with the **tabulated phase function**.

    Parameters
    ----------
    lam : (nλ,) array-like of float
        Wavelengths [µm], strictly positive.
    tfilm : float
        Film thickness [mm], ≥ 0.
    fv : float
        Particle volume fraction (0 ≤ f_v < 1).
    diameters : (nD,) array-like of float
        Sphere diameters [µm], strictly positive.
    size_dist : (nD,) array-like of float
        **Number-fraction** weights p_i (Case A). Will be renormalized to sum to 1 if slightly off.
    Np : complex or (nλ,) array-like of complex
        Particle refractive index.
    Nh : complex or (nλ,) array-like of complex
        Host refractive index.
    Nup, Ndw : complex or (nλ,) array-like of complex, optional
        Superstrate/substrate refractive indices. Default 1.0 (air).
    effective_medium : bool, optional (default True)
        If True, computes an effective host refractive index via the Bruggeman

    Returns
    -------
    Ttot, Rtot, Tspec, Rspec : (nλ,) ndarrays
        Total and specular transmittance/reflectance at normal incidence.

    Notes
    -----
    - Units: μ_* returned to `adm` are in µm⁻¹, as expected by your `adm` wrapper.
    - **Do not** multiply by ⟨A⟩ here: ⟨C⟩ already includes area; coefficients are n_tot·⟨C⟩.
    """
    # ---------- coerce arrays & basic checks ----------
    lam = _np.asarray(lam, float).ravel()
    if lam.ndim != 1 or lam.size == 0 or _np.any(lam <= 0) or not _np.all(_np.isfinite(lam)):
        raise ValueError("lam must be a 1D array of positive finite wavelengths [µm].")
    nlam = lam.size

    D = _np.asarray(diameters, float).ravel()
    p = _np.asarray(size_dist, float).ravel()

    if not (0 <= float(fv) < 1):
        raise ValueError("fv (volume fraction) must be in [0,1).")
    if not _np.isscalar(tfilm) or tfilm < 0:
        raise ValueError("tfilm must be a nonnegative scalar in mm.")

    # ---------- call Mie polydisperse helper ----------
    csca_av, cabs_av, _, phase_scatter = mie.poly_sphere_cross_section(
            lam, D, p, Np, Nh, fv, effective_medium=effective_medium)

    # ---------- n_tot and coefficients (µm⁻¹) ----------
    Ac = _np.pi * (D / 2.0) ** 2      # [µm²] (not used in formula below; kept for clarity)
    V  = (4.0 / 3.0) * _np.pi * (D / 2.0) ** 3  # [µm³]
    V_mean = float(_np.sum(p * V))    # ⟨V⟩ [µm³]
    if not _np.isfinite(V_mean) or V_mean <= 0:
        raise RuntimeError("Average particle volume ⟨V⟩ must be positive/finite.")

    n_tot = fv / V_mean              # [µm⁻³]
    k_sca = n_tot * csca_av          # [µm⁻¹]
    k_abs = n_tot * cabs_av          # [µm⁻¹]

    # ---------- radiative transfer ----------
    Ttot, Rtot, Tspec, Rspec = adm(
            lam, tfilm, k_sca, k_abs, Nh, phase_fun=phase_scatter, Nup=Nup, Ndw=Ndw)

    return Ttot, Rtot, Tspec, Rspec

def adm(lam, tfilm, k_sca, k_abs, Nh,
    gcos=None,            # optional: asymmetry parameter per λ
    *,
    phase_fun=None,       # optional: phase function vs angle or mu for each λ
    theta_deg=None,       # required if phase_fun is given as θ-grid (degrees)
    Nup=1.0,              # refractive index above
    Ndw=1.0,              # refractive index below
    quad_pts: int = 32,   # IAD quadrature points when using a tabulated PF
):
    """
    Adding–doubling (IAD) reflectance/transmittance for a scattering/absorbing film.

    Inputs (arrays are per-wavelength unless noted):
    - lam      : (nλ,) wavelengths [µm]
    - tfilm    : scalar film thickness [mm]
    - k_sca    : (nλ,) scattering coefficient [µm^-1]
    - k_abs    : (nλ,) absorption coefficient [µm^-1]
    - Nh       : scalar or (nλ,) complex refractive index of the film host
    --------------------------------------------------------------------------
    Choose ONE angular description:
    - gcos     : (nλ,) asymmetry parameter  (classic Henyey–Greenstein style use)
      OR
    - phase_fun: tabulated differential *phase function* (not normalized to 1/4π),
                 shape (nθ, nλ) ndarray or DataFrame. If rows are θ[deg], pass `theta_deg`.
                 If index is cosθ (μ), set `theta_deg=None`.
    - theta_deg: (nθ,) angles in degrees corresponding to phase_fun rows (if needed)
    --------------------------------------------------------------------------
    - Nup, Ndw : scalar or (nλ,) complex refractive indices above/below (defaults=1.0)
    - quad_pts : number of quadrature points for IAD when using a tabulated phase function (default=32)

    Returns:
    - Ttot, Rtot, Tspec, Rspec : each (nλ,) arrays
    """
    # ---------- coerce arrays ----------
    lam   = _np.asarray(lam,   float)
    k_sca = _np.asarray(k_sca, float)
    k_abs = _np.asarray(k_abs, float)

    if lam.ndim != 1:
        raise ValueError("lam must be a 1D array of wavelengths [µm].")
    nlam = lam.size
    for name, arr in [("k_sca", k_sca), ("k_abs", k_abs)]:
        if _np.asarray(arr).shape != (nlam,):
            raise ValueError(f"{name} must have the same length as lam.")

    Nh_arr  = as_carray(Nh,  "Nh", nlam, val_type=complex)
    Nup_arr = as_carray(Nup, "Nup", nlam, val_type=complex)
    Ndw_arr = as_carray(Ndw, "Ndw", nlam, val_type=complex)

    # ---------- convert to IAD units (mm^-1); include host absorption via Im(n) ----------
    # k_sca, k_abs are in µm^-1 -> mm^-1 multiply by 1e3
    mu_s = k_sca * 1e3
    # add material absorption from host's Im(n): kz_imag = (2π/λ)*Im(n) in mm^-1 (λ in µm -> factor 1e3)
    kz_imag = 2.0 * _np.pi / lam * Nh_arr.imag * 1e3
    mu_a = k_abs * 1e3 + 2.0 * kz_imag
    mu_t = mu_s + mu_a
    d = float(tfilm)

    # ---------- choose angular description ----------
    use_pf = phase_fun is not None
    if use_pf and (gcos is not None):
        raise ValueError("Provide either gcos OR phase_fun, not both.")
    if (not use_pf) and (gcos is None):
        raise ValueError("You must provide gcos (per λ) or a tabulated phase_fun.")

    if not use_pf:
        gcos = _np.asarray(gcos, float)
        if gcos.shape != (nlam,):
            raise ValueError("gcos must have shape (len(lam),).")

    # ---------- prepare phase function (optional branch) ----------
    pf_cols = None
    pf_idx  = None
    if use_pf:
        if isinstance(phase_fun, pd.DataFrame):
            pf_cols = _np.asarray(phase_fun.columns)
            pf_idx  = _np.asarray(phase_fun.index)
            pf_vals = phase_fun.values  # (nθ, nλ')
        else:
            pf_vals = _np.asarray(phase_fun, float)
            if pf_vals.ndim != 2:
                raise ValueError("phase_fun must be 2D (nθ, nλ).")
        # build (nθ, nλ) array aligned to lam
        if isinstance(phase_fun, pd.DataFrame):
            # If DataFrame columns are numeric wavelengths, align to lam
            if pf_cols.shape == (nlam,) and _np.allclose(pf_cols.astype(float), lam, rtol=1e-6, atol=1e-6):
                PF = pf_vals
            else:
                # try to reindex with nearest match
                try:
                    PF = phase_fun.reindex(columns=lam, method=None).values
                except Exception as e:
                    raise ValueError("phase_fun columns must match lam (wavelengths).") from e
        else:
            if pf_vals.shape[1] != nlam:
                raise ValueError("phase_fun shape must be (nθ, nλ) matching len(lam).")
            PF = pf_vals

        # index handling: if theta_deg given, convert to μ = cosθ and ensure ascending μ
        if theta_deg is not None:
            theta_deg = _np.asarray(theta_deg, float)
            if theta_deg.shape != (PF.shape[0],):
                raise ValueError("theta_deg length must match phase_fun rows.")
            mu = _np.cos(_np.radians(theta_deg))
        else:
            # try to interpret given index as μ
            if pf_idx is None:
                raise ValueError("phase_fun given as ndarray requires theta_deg.")
            mu = _np.asarray(pf_idx, float)

        order = _np.argsort(mu)  # ascending μ in [-1,1]
        mu = mu[order]
        PF = PF[order, :]

        # IAD expects a DataFrame with index μ in [-1,1], one column per λ
        pf_df = pd.DataFrame(PF, index=mu, columns=lam)
        pf_df.index.name = "cos(theta)"

    # ---------- run IAD per wavelength ----------
    Ttot = _np.zeros(nlam, float)
    Rtot = _np.zeros(nlam, float)
    Tspec = _np.zeros(nlam, float)
    Rspec = _np.zeros(nlam, float)

    for j in range(nlam):
        # guard: IAD wants n >= 1
        n_real = float(max(Nh_arr.real[j], 1.0))
        n_up   = float(max(Nup_arr.real[j], 1.0))
        n_dw   = float(max(Ndw_arr.real[j], 1.0))

        if mu_t[j] <= 0.0:
            # transparent, non-scattering layer -> Fresnel only
            s = _iad.Sample(a=0.0, b=0.0, g=0.0, d=d, n=n_real, n_above=n_up, n_below=n_dw)
        else:
            a = mu_s[j] / mu_t[j]     # single-scattering albedo
            b = mu_t[j] * d           # optical thickness
            if not use_pf:
                s = _iad.Sample(a=a, b=b, g=float(gcos[j]), d=d,
                               n=n_real, n_above=n_up, n_below=n_dw)
            else:
                # tabulated phase function path
                # IAD expects one column for the current wavelength
                pf_col = pf_df.iloc[:, j].to_frame()
                s = _iad.Sample(a=a, b=b, d=d, n=n_real, n_above=n_up, n_below=n_dw,
                               quad_pts=int(quad_pts),
                               pf_type="TABULATED", pf_data=pf_col)

        # normal-incidence total RT
        R_tot, T_tot, _, _ = s.rt()
        # unscattered (specular) RT
        R_spec, T_spec = s.unscattered_rt()

        Ttot[j]  = float(T_tot)
        Rtot[j]  = float(R_tot)
        Tspec[j] = float(T_spec)
        Rspec[j] = float(R_spec)

    return Ttot, Rtot, Tspec, Rspec
 
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
from . import nklib as nk
import iadpython as _iad
import pandas as pd
from typing import Union as _Union, Optional as _Optional, List as _List
from .utils import as_carray, _check_mie_inputs
from .nklib import emt_brugg, emt_multilayer_sphere

def T_beer_lambert(lam: _Union[float, _np.ndarray],                                         # wavelengths [µm]
                   Nh: _Union[float, _np.ndarray],                                     # host refractive index
                   Np: _Union[float, _np.ndarray, _List[_Union[float, _np.ndarray]]],  # particle refractive index
                   D: _Union[float, _np.ndarray, _List[_Union[float, _np.ndarray]]],   # sphere diameters [µm] 
                   fv: float,                                                          # film volume fraction 
                   tfilm: float, 
                   *,
                   theta: _Union[float, _np.ndarray]= 0.0,                            # angle of incidence in degrees
                   Nup: _Union[float, _np.ndarray] = 1.0,                              # refractive index above
                   Ndw: _Union[float, _np.ndarray] = 1.0,                              # refractive index below
                   size_dist: _np.ndarray = None,                                      # number-fraction weights p_i 
                   dependent_scatt = False,                                            # use Perkus-Yevik for dependent scattering
                   effective_medium: bool = True,                                      # whether to compute effective Nh via Bruggeman
                   use_phase_fun: bool = False,                                        # whether to use phase function instead of g
                   check_inputs = True                                                 # whether to check mie inputs
                   ):
    '''
    Transmittance and reflectance from Beer-Lamberts law for a film with 
    spherical particles. Reflectance is cokmputed from classical formulas for 
    incoherent light incident on a slab between two semi-infinite media 
    (no scattering is considered for this parameter)

 Parameters
    ----------
    lam : array-like, shape (nλ,)
        Wavelengths [µm], strictly positive.

    Nh : float or array-like (nλ,)
        Host refractive index (can be complex). If array-like, length must equal len(lam).

    Np (float, 1darray or list): Complex refractive index of each 
                                            shell layer. Np.shape[1] == len(D). 
        Options are:
        float:   solid sphere and constant refractive index
        1darray: solid sphere and spectral refractive index (len = lam)
        list:    multilayered sphere (with both constant or spectral refractive indexes)
    
    D : float, _np.ndarray or list
        Diameter of the spheres. Use float for monodisperse, or array for polydisperse.
        if multilayer sphere, use list of floats (monodisperse) or arrays (polydisperse).
    
    fv : float
        Particle volume fraction in (0, 1). Used only to compute an effective medium Nh via
        `nk.emt_brugg(fv, Np, Nh)`.

    tfilm : float
        Film thickness [mm], ≥ 0.
            
    Nup : float or array-like, optional
        Refractive index above the film. Default is 1.0 (air). If array-like, length must equal len(lam).

    Ndw : float or array-like, optional
        Refractive index below the film. Default is 1.0 (air). If array-like, length must equal len(lam).

    size_dist : array-like, shape (n_bins,), optional
        Number-fraction weights for polydisperse particles. Default is None (monodisper
        particles). If given, must be 1D array with len(size_dist) == len(D[0]) (if D is list)
        or len(size_dist) == len(D) (if D is array).
        The weights must be nonnegative and sum to 1.
        The size distribution is used only when `dependent_scatt=True`.
    
    dependent_scatt : bool, optional
        Whether to include dependent scattering effects via Percus-Yevick structure factor
        (default: False; not recommended for metallic spheres or high fv)
    
    effective_medium : bool, optional
        Whether to compute an effective medium for the host refractive index via Bruggeman
        (default: True; recommended for fv >~ 0.1)
    
    use_phase_fun : bool, optional
        Whether to use the full phase function in the radiative transfer (default: False).
        If False, the asymmetry parameter g is used instead (Henyey-Greenstein approximation).
        Using the phase function is more accurate but also more computationally intensive.
    
    check_inputs : bool, optional
        Whether to check mie inputs (default: True)    
    Returns
    -------
    Ttot : array-like, shape (nλ,)
        Total transmittance.
    
    Rtot : array-like, shape (nλ,)
        Total reflectance.
    
    Tspec : array-like, shape (nλ,)
        Specular transmittance.
    
    Rspec : array-like, shape (nλ,)
        Specular reflectance.
    '''

    # ---------- coerce arrays & basic checks ----------
    if check_inputs:
        lam, Nh, Np, D, size_dist = _check_mie_inputs(lam, Nh, Np, D, size_dist=size_dist)

    nlam = lam.size
    Nup = as_carray(Nup, "Nup", nlam, val_type=complex)
    Ndw = as_carray(Ndw, "Ndw", nlam, val_type=complex)

    if not (0 <= float(fv) < 1):
        raise ValueError("fv (volume fraction) must be in [0,1).")
    if not _np.isscalar(tfilm) or tfilm < 0:
        raise ValueError("tfilm must be a nonnegative scalar in mm.")

    # if dependent scatt, check that particle is metallic through: Im(Np) > Re(Np)
    if dependent_scatt:
        if _np.any(_np.array(Np).real < _np.array(Np).imag):
            print("Warning: Dependent scattering theory not recommended for metallic particles.")
    
    # ---------- Effective medium for host (if your convention is to dress Nh) ----------
    N_layers = len(D)                                    # number of layers in the sphere
    if effective_medium:
        # Compute mean diameter of each layer
        D_layers_mean = []
        for i in range(N_layers):
            if size_dist is None:
                # Monodisperse
                D_layers_mean.append(D[i])  # -> float
            else:
                # Polydisperse
                D_layers_mean.append(_np.average(D[i], axis=0,   # -> float
                                            weights=size_dist))  # size_dist shape (n_bins,)

        # Compute effective refractive index of host using Bruggeman EMT                                   
        Np_eff = emt_multilayer_sphere(D_layers_mean, Np, check_inputs=False)
        Nh = emt_brugg(fv, Np_eff, Nh)

    # ---------- Mie cross sections and phase function ----------
    csca, cabs, _, _ = mie.cross_section_ensemble(lam, D, size_dist, Np, Nh, fv, 
                                                  size_dist=size_dist,
                                                  check_inputs=False,
                                                  effective_medium=False,
                                                  dependent_scatt=dependent_scatt,
                                                  phase_function=use_phase_fun)

    # ---------- n_tot and coefficients (µm⁻¹) ----------
    # Particle volume (or mean volume if polydisperse)
    V  = (4.0 / 3.0) * _np.pi * (D[-1] / 2.0) ** 3  # [µm³]
    if size_dist is not None:
            V = float(_np.sum(size_dist * V))              # ⟨V⟩ [µm³]

    # Get scattering and absorption coefficients
    n_tot = fv / V                # [µm⁻³]
    k_sca = n_tot * csca          # [µm⁻¹]
    k_abs = n_tot * cabs          # [µm⁻¹]
    k_ext = k_sca + k_abs         # [µm⁻¹]

    # ---------- Fresnel reflectance/transmittance ----------
    tfilm = tfilm*1E3 # convert mm to micron units
    
    Rp, Tp = wv.incoh_multilayer(lam, theta, (Nup, Nh, Ndw), tfilm, pol = 'TM')
    Rs, Ts = wv.incoh_multilayer(lam, theta, (Nup, Nh, Ndw), tfilm, pol = 'TE')
    T    = (Ts + Tp)/2
    Rtot = (Rp + Rs)/2
    
    theta1 = wv.snell(Nup,Nh, theta)
        
    Ttot = T*_np.exp(-k_abs*tfilm/_np.cos(theta1.real))
    Tspec = T*_np.exp(-k_ext*tfilm/_np.cos(theta1.real))

    return Ttot, Rtot, Tspec

def adm_sphere(lam: _Union[float, _np.ndarray],                                     # wavelengths [µm]
                Nh: _Union[float, _np.ndarray],                                     # host refractive index
                Np: _Union[float, _np.ndarray, _List[_Union[float, _np.ndarray]]],  # particle refractive index
                D: _Union[float, _np.ndarray, _List[_Union[float, _np.ndarray]]],   # sphere diameters [µm] 
                fv: float,                                                          # film volume fraction 
                tfilm: float,                                                       # film thickness [mm]       
                *,
                Nup: _Union[float, _np.ndarray] = 1.0,                              # refractive index above
                Ndw: _Union[float, _np.ndarray] = 1.0,                              # refractive index below
                size_dist: _np.ndarray = None,                                      # number-fraction weights p_i 
                dependent_scatt = False,                                            # use Perkus-Yevik for dependent scattering
                effective_medium: bool = True,                                      # whether to compute effective Nh via Bruggeman
                use_phase_fun: bool = False,                                        # whether to use phase function instead of g
                check_inputs = True                                                 # whether to check mie inputs
                ):
    '''
    Parameters
    ----------
    lam : array-like, shape (nλ,)
        Wavelengths [µm], strictly positive.

    Nh : float or array-like (nλ,)
        Host refractive index (can be complex). If array-like, length must equal len(lam).

    Np (float, 1darray or list): Complex refractive index of each 
                                            shell layer. Np.shape[1] == len(D). 
        Options are:
        float:   solid sphere and constant refractive index
        1darray: solid sphere and spectral refractive index (len = lam)
        list:    multilayered sphere (with both constant or spectral refractive indexes)
    
    D : float, _np.ndarray or list
        Diameter of the spheres. Use float for monodisperse, or array for polydisperse.
        if multilayer sphere, use list of floats (monodisperse) or arrays (polydisperse).
    
    fv : float
        Particle volume fraction in (0, 1). Used only to compute an effective medium Nh via
        `nk.emt_brugg(fv, Np, Nh)`.

    tfilm : float
        Film thickness [mm], ≥ 0.
            
    Nup : float or array-like, optional
        Refractive index above the film. Default is 1.0 (air). If array-like, length must equal len(lam).

    Ndw : float or array-like, optional
        Refractive index below the film. Default is 1.0 (air). If array-like, length must equal len(lam).

    size_dist : array-like, shape (n_bins,), optional
        Number-fraction weights for polydisperse particles. Default is None (monodisper
        particles). If given, must be 1D array with len(size_dist) == len(D[0]) (if D is list)
        or len(size_dist) == len(D) (if D is array).
        The weights must be nonnegative and sum to 1.
        The size distribution is used only when `dependent_scatt=True`.
    
    dependent_scatt : bool, optional
        Whether to include dependent scattering effects via Percus-Yevick structure factor
        (default: False; not recommended for metallic spheres or high fv)
    
    effective_medium : bool, optional
        Whether to compute an effective medium for the host refractive index via Bruggeman
        (default: True; recommended for fv >~ 0.1)
    
    use_phase_fun : bool, optional
        Whether to use the full phase function in the radiative transfer (default: False).
        If False, the asymmetry parameter g is used instead (Henyey-Greenstein approximation).
        Using the phase function is more accurate but also more computationally intensive.
    
    check_inputs : bool, optional
        Whether to check mie inputs (default: True)    
    Returns
    -------
    Ttot : array-like, shape (nλ,)
        Total transmittance.
    
    Rtot : array-like, shape (nλ,)
        Total reflectance.
    
    Tspec : array-like, shape (nλ,)
        Specular transmittance.
    
    Rspec : array-like, shape (nλ,)
        Specular reflectance.
    '''

    # ---------- coerce arrays & basic checks ----------
    if check_inputs:
        lam, Nh, Np, D, size_dist = _check_mie_inputs(lam, Nh, Np, D, size_dist=size_dist)

    nlam = lam.size
    Nup = as_carray(Nup, "Nup", nlam, val_type=complex)
    Ndw = as_carray(Ndw, "Ndw", nlam, val_type=complex)

    if not (0 <= float(fv) < 1):
        raise ValueError("fv (volume fraction) must be in [0,1).")
    if not _np.isscalar(tfilm) or tfilm < 0:
        raise ValueError("tfilm must be a nonnegative scalar in mm.")

    # if dependent scatt, check that particle is metallic through: Im(Np) > Re(Np)
    if dependent_scatt:
        if _np.any(_np.array(Np).real < _np.array(Np).imag):
            print("Warning: Dependent scattering theory not recommended for metallic particles.")

    # ---------- Effective medium for host (if your convention is to dress Nh) ----------
    N_layers = len(D)                                    # number of layers in the sphere
    if effective_medium:
        # Compute mean diameter of each layer
        D_layers_mean = []
        for i in range(N_layers):
            if size_dist is None:
                # Monodisperse
                D_layers_mean.append(D[i])  # -> float
            else:
                # Polydisperse
                D_layers_mean.append(_np.average(D[i], axis=0,   # -> float
                                            weights=size_dist))  # size_dist shape (n_bins,)

        # Compute effective refractive index of host using Bruggeman EMT                                   
        Np_eff = emt_multilayer_sphere(D_layers_mean, Np, check_inputs=False)
        Nh = emt_brugg(fv, Np_eff, Nh)

    # ---------- Mie cross sections and phase function ----------
    csca, cabs, gcos, phase_scatter = mie.cross_section_ensemble(lam, D, size_dist, Np, Nh, fv, 
                                                                size_dist=size_dist,
                                                                check_inputs=False,
                                                                effective_medium=False,
                                                                dependent_scatt=dependent_scatt,
                                                                phase_function=use_phase_fun)

    # ---------- n_tot and coefficients (µm⁻¹) ----------
    # Particle volume (or mean volume if polydisperse)
    V  = (4.0 / 3.0) * _np.pi * (D[-1] / 2.0) ** 3  # [µm³]
    if size_dist is not None:
            V = float(_np.sum(size_dist * V))              # ⟨V⟩ [µm³]

    # Get scattering and absorption coefficients
    n_tot = fv / V                # [µm⁻³]
    k_sca = n_tot * csca          # [µm⁻¹]
    k_abs = n_tot * cabs          # [µm⁻¹]

    # ---------- radiative transfer ----------
    if use_phase_fun:
        Ttot, Rtot, Tspec, Rspec = adm(lam, tfilm, k_sca, k_abs, Nh, 
                                       phase_fun=phase_scatter, 
                                       Nup=Nup, 
                                       Ndw=Ndw)
    else:
        Ttot, Rtot, Tspec, Rspec = adm(lam, tfilm, k_sca, k_abs, Nh, 
                                       gcos=gcos, 
                                       Nup=Nup, 
                                       Ndw=Ndw)

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

    Nh_arr  = as_carray(Nh,  "Nh" , nlam, val_type=complex)
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
    Ttot  = _np.zeros(nlam, float)
    Rtot  = _np.zeros(nlam, float)
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
 
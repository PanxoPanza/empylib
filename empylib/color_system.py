# color_system.py
# -*- coding: utf-8 -*-
"""
This library contains the class ColorSystem that converts a spectra into RGB colors
Created on Wed 17 Aug, 2022

@author: PanxoPanza
"""

import numpy as np
import colour as clr
from typing import Tuple

# ---------------------------- helpers ---------------------------------

def _material_rgb_from_factor(
    w_nm: np.ndarray,
    F: np.ndarray,
    *,
    illuminant,
    illuminant_name: str,
    illuminant_units: str,
    observer_name: str,
    interval_nm: float,
) -> np.ndarray:
    """
    Material path (R/T): integrate F(λ)*Illuminant(λ)*CMFs(λ),
    adapt to sRGB (D65), encode. Returns sRGB in [0,1].
    """
    # 1) Build illuminant SD (accept string name, tuple (wls_um, spd), or SpectralDistribution)
    if illuminant is None:
        sd_Ill = clr.SDS_ILLUMINANTS[illuminant_name]
        xy_in_named = clr.CCS_ILLUMINANTS[observer_name][illuminant_name]
        xy_in = np.array(xy_in_named, float)
    elif isinstance(illuminant, clr.SpectralDistribution):
        sd_Ill = illuminant
        xy_in = None  # compute later from SD
    else:
        ill_wls_um, ill_spd = illuminant
        ill_wls_um = np.asarray(ill_wls_um, float).ravel()
        ill_spd    = np.asarray(ill_spd, float).ravel()
        if ill_wls_um.size != ill_spd.size:
            raise ValueError("Custom illuminant wavelengths and SPD must have same length.")
        # μm→nm; make SPD per-nm for integration on an nm grid
        w_nm_ill = ill_wls_um * 1000.0
        if illuminant_units.lower() == "per_um":
            ill_spd = ill_spd / 1000.0
        elif illuminant_units.lower() != "per_nm":
            raise ValueError("illuminant_units must be 'per_um' or 'per_nm'.")
        sd_Ill = clr.SpectralDistribution(dict(zip(w_nm_ill, ill_spd)), name="custom illuminant")
        xy_in = None  # compute later from SD

    # 2) CMFs and intersection grid
    cmfs = clr.MSDS_CMFS[observer_name]
    start = max(min(w_nm), sd_Ill.shape.start, cmfs.shape.start)
    end   = min(max(w_nm), sd_Ill.shape.end,   cmfs.shape.end)
    if not (end > start):
        raise ValueError("No spectral overlap between sample, illuminant, and CMFs.")
    shape = clr.SpectralShape(start, end, interval_nm)

    sd_F   = clr.SpectralDistribution(dict(zip(w_nm, F)), name="material").interpolate(shape).align(shape)
    sd_Ill = sd_Ill.copy().interpolate(shape).align(shape)
    cmfs   = cmfs.copy().interpolate(shape).align(shape)

    # 3) Spectrum -> XYZ under chosen illuminant/observer (Y=100 for perfect diffuser)
    XYZ = clr.sd_to_XYZ(sd_F, cmfs=cmfs, illuminant=sd_Ill) / 100.0

    # 4) Input whitepoint for CAT02:
    #    named -> already known; custom -> compute xy from illuminant SPD and CMFs
    if xy_in is None:
        XYZ_wp = clr.sd_to_XYZ(sd_Ill, cmfs=cmfs)   # treat illuminant as an emitter
        xy_in  = clr.XYZ_to_xy(XYZ_wp)

    # 5) XYZ -> sRGB (D65)
    srgb = clr.RGB_COLOURSPACES["sRGB"]
    rgb = clr.XYZ_to_RGB(
        XYZ,
        srgb,
        illuminant=xy_in,                      # adapt from input white to D65
        chromatic_adaptation_transform="CAT02",
        apply_cctf_encoding=True,
    )
    return np.clip(rgb, 0.0, 1.0)

def _emitter_rgb_from_spd(
    w_nm: np.ndarray,
    S_per_nm: np.ndarray,
    *,
    observer_name: str,
    interval_nm: float,
) -> np.ndarray:
    """
    Emitter path: integrate SPD(λ)*CMFs(λ), normalise to Y=1, sRGB encode.
    No illuminant and no chromatic adaptation. Returns sRGB in [0,1].
    """
    sd_S = clr.SpectralDistribution(dict(zip(w_nm, S_per_nm)), name="emitter")
    cmfs = clr.MSDS_CMFS[observer_name]

    start = max(min(w_nm), cmfs.shape.start)
    end   = min(max(w_nm), cmfs.shape.end)
    if not (end > start):
        raise ValueError("No spectral overlap between emitter SPD and CMFs.")
    shape = clr.SpectralShape(start, end, interval_nm)

    sd_S = sd_S.copy().interpolate(shape).align(shape)
    cmfs = cmfs.copy().interpolate(shape).align(shape)

    # ---- changed lines: use .domain and .values (not .items) ---------------
    w = np.asarray(sd_S.domain, float)     # wavelength grid [nm]
    S = np.asarray(sd_S.values, float)     # SPD per nm on that grid
    cm = np.asarray(cmfs.values, float)    # columns: x̄, ȳ, z̄

    X = np.trapz(S * cm[:, 0], w)
    Y = np.trapz(S * cm[:, 1], w)
    Z = np.trapz(S * cm[:, 2], w)
    if Y <= 0:
        return np.zeros(3)

    XYZ = np.array([X, Y, Z]) / Y
    srgb = clr.RGB_COLOURSPACES["sRGB"]
    rgb = clr.XYZ_to_RGB(
        XYZ, srgb,
        chromatic_adaptation_transform=None,
        apply_cctf_encoding=True,
    )
    return np.clip(rgb, 0.0, 1.0)


# ----------------------------- API ------------------------------------

def spectrum_to_hex(
    wls_um: np.ndarray,
    values: np.ndarray,
    *,
    source: str = "material",                 # "material" (R/T/1-A) or "emitter"
    illuminant=None,                          # None | (ill_wls_um, ill_spd) | colour.SpectralDistribution
    illuminant_name: str = "D65",             # used when illuminant is None
    illuminant_units: str = "per_um",         # only for custom tuple: "per_nm" or "per_um"
    observer_name: str = "CIE 1931 2 Degree Standard Observer",
    interval_nm: float = 1.0,
    emitter_units: str = "per_um",            # only for source="emitter"
) -> Tuple[str, Tuple[float, float, float], Tuple[int, int, int]]:
    """
    Convert a spectrum into an sRGB colour using standard CIE colorimetry.

    The function supports two physically different cases:

    - `source="material"`: `values` is a *spectral factor* F(λ) in [0,1] (reflectance, transmittance,
      or equivalently 1 - absorptance). The function integrates F(λ) under a viewing illuminant
      (named or custom), computes CIE XYZ for the chosen observer, chromatically adapts from the
      illuminant’s white to sRGB’s D65, and encodes to sRGB.
    - `source="emitter"`: `values` is a *spectral power distribution* S(λ) of a self-luminous source.
      The function integrates S(λ) directly with the CIE CMFs (no illuminant, no adaptation) and
      encodes to sRGB. The absolute scale of S(λ) does not affect the resulting colour (we normalise).

    Parameters
    ----------
    wls_um : np.ndarray, shape (N,)
        Wavelength grid in micrometres (μm). Can be unsorted; it will be sorted ascending.
        Must be finite numeric values.

    values : np.ndarray, shape (N,)
        Spectrum sampled at `wls_um`.
        - If `source="material"`: dimensionless factors in [0,1]. Values outside are clipped.
        - If `source="emitter"`: spectral power density (relative scale OK). Units are given by
          `emitter_units`.

    source : {"material", "emitter"}, keyword-only
        Selects the computation path:
        - `"material"` → F(λ) × Illuminant(λ) × CMFs → XYZ → CAT02→D65 → sRGB.
        - `"emitter"`  → SPD(λ) × CMFs → XYZ (normalise Y=1) → sRGB (no CAT).

    illuminant : None | (np.ndarray, np.ndarray) | colour.SpectralDistribution
        Illuminant to use when `source="material"`.
        - `None` → use `illuminant_name` from the internal library (e.g., "D65", "A", "D50", "F11").
        - `(ill_wls_um, ill_spd)` → custom illuminant SPD. `ill_wls_um` in μm; `ill_spd` density given
          by `illuminant_units`. Scale is irrelevant to chromaticity; shape matters.
        - `colour.SpectralDistribution` → already-built illuminant SD (assumed per-nm density).

    illuminant_name : str
        Name of a built-in illuminant used when `illuminant is None`. Typical choices: "D65"
        (default, matches sRGB), "D50" (printing), "A" (tungsten), "D55"/"D60"/"D75", "F1"… "F12".

    illuminant_units : {"per_um", "per_nm"}
        Units for a *custom* illuminant SPD passed as a tuple. If `"per_um"`, the SPD is divided by
        1000 to convert to per-nm before integration.

    observer_name : str
        Colour-matching functions (CMFs) dataset. Commonly
        `"CIE 1931 2 Degree Standard Observer"` (default) or
        `"CIE 1964 10 Degree Standard Observer"`. Must exist in `colour.MSDS_CMFS`.

    interval_nm : float
        Resampling step (nm) of the common spectral grid used for integration. Use 1 nm for accuracy,
        ≤5 nm for speed. Must be > 0.

    emitter_units : {"per_um", "per_nm"}
        Units of the emitter SPD when `source="emitter"`. If `"per_um"`, it is divided by 1000 so the
        numeric integral over nm is correct.

    Returns
    -------
    hex_color : str
        HTML hex colour, e.g. "#6096ff".

    rgb01 : Tuple[float, float, float]
        sRGB components as floats in [0,1], **including** the sRGB transfer function (gamma).

    rgb255 : Tuple[int, int, int]
        sRGB components quantised to 8-bit integers [0,255].

    Raises
    ------
    ValueError
        - `interval_nm <= 0`
        - `wls_um.size != values.size`
        - `emitter_units` or `illuminant_units` not in {"per_um","per_nm"}
        - `source` not in {"material","emitter"}
        - (From helpers) no spectral overlap between sample/illuminant/CMFs.

    TypeError
        - `wls_um` or `values` is not a NumPy array.

    Notes
    -----
    Algorithm outline (what happens internally):

    1) **Validate & sort.** Wavelengths (μm) and values are sorted; wavelengths converted to nm.
    2) **Branch by `source`.**
       - **Emitter:** convert SPD to per-nm if needed → intersect with CMFs domain →
         integrate `∫ S(λ)·CMF(λ) dλ` → obtain XYZ → normalise to Y=1 (so absolute power cancels) →
         convert XYZ→sRGB (no chromatic adaptation) and apply the sRGB OETF.
       - **Material:** clip F(λ) to [0,1] → build illuminant (named or custom) →
         intersect sample, illuminant, CMFs → compute XYZ via `sd_to_XYZ` (Y=100 for a perfect
         diffuser) → compute input whitepoint from the illuminant (named or from its SPD) →
         chromatically adapt XYZ to D65 (CAT02) → convert to sRGB and apply OETF.
    3) **Clamp & pack.** Clip sRGB to [0,1], then form hex and 8-bit tuples.

    Practical guidance:
    - Use `source="material", illuminant_name="D65"` for materials viewed on a standard display.
    - Use a **custom illuminant** (e.g., AM1.5G) when simulating outdoor/solar conditions:
      pass `(lam_um, E(λ))` and set `illuminant_units` appropriately.
    - For **emitters** (blackbody, LEDs, lasers), pass the SPD and the correct `emitter_units`;
      the hue/saturation are independent of the absolute scale.

    Examples
    --------
    # 1) Material under D65
    hex_color, rgb01, rgb255 = spectrum_to_hex(lam_um, R_lambda, source="material")

    # 2) Material under custom AM1.5G
    hex_color, *_ = spectrum_to_hex(
        lam_um, R_lambda,
        source="material",
        illuminant=(lam_solar_um, E_AM15G_per_nm),
        illuminant_units="per_nm"
    )

    # 3) Blackbody emitter (B_λ per μm)
    Ebb = np.pi * Bplanck(lam_um, T)  # W·m^-2·μm^-1
    hex_color, *_ = spectrum_to_hex(
        lam_um, Ebb,
        source="emitter",
        emitter_units="per_um"
    )
    """
    if interval_nm <= 0:
        raise ValueError("interval_nm must be > 0.")
    if not isinstance(wls_um, np.ndarray) or not isinstance(values, np.ndarray):
        raise TypeError("wls_um and values must be numpy arrays.")
    if wls_um.size != values.size:
        raise ValueError("wls_um and values must have the same length.")

    # sort and convert μm → nm
    idx = np.argsort(wls_um.astype(float))
    w_nm = (wls_um[idx].astype(float) * 1000.0)
    vals = values[idx].astype(float)

    s = source.lower()
    if s == "emitter":
        if emitter_units.lower() == "per_um":
            vals = vals / 1000.0
        elif emitter_units.lower() != "per_nm":
            raise ValueError("emitter_units must be 'per_um' or 'per_nm'.")
        rgb = _emitter_rgb_from_spd(
            w_nm, vals,
            observer_name=observer_name,
            interval_nm=interval_nm,
        )
    elif s == "material":
        vals = np.clip(vals, 0.0, 1.0)
        rgb = _material_rgb_from_factor(
            w_nm, vals,
            illuminant=illuminant,
            illuminant_name=illuminant_name,
            illuminant_units=illuminant_units,
            observer_name=observer_name,
            interval_nm=interval_nm,
        )
    else:
        raise ValueError("source must be 'material' or 'emitter'.")

    rgb255 = (rgb * 255.0 + 0.5).astype(int)
    hex_color = f"#{rgb255[0]:02x}{rgb255[1]:02x}{rgb255[2]:02x}"
    return hex_color, (float(rgb[0]), float(rgb[1]), float(rgb[2])), (int(rgb255[0]), int(rgb255[1]), int(rgb255[2]))

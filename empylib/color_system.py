# color_system.py
# -*- coding: utf-8 -*-
"""
This library contains the class ColorSystem that converts a spectra into RGB colors
Created on Wed 17 Aug, 2022

@author: PanxoPanza
"""

import numpy as np
import colour as clr
from typing import Sequence, Tuple

def spectrum_to_hex(
    wls_um: Sequence[float],
    R: Sequence[float],
    illuminant_name: str = "D65",
    observer_name: str = "CIE 1931 2 Degree Standard Observer",
    interval_nm: float = 1.0,
) -> Tuple[str, Tuple[float, float, float], Tuple[int, int, int]]:
    """
    Convert a reflectance/transmittance spectrum to an sRGB color (HEX + RGB), using proper CIE colorimetry.

    Parameters
    ----------
    wls_um: ndarray, shape (N,)
        1D wavelengths in micrometers (μm). They will be sorted ascending and converted to nm.
    R: ndarray, shape (N,)
        1D *reflectance* values in [0..1], same length as wls_um. If values fall outside [0..1],
        they are clipped (fluorescent samples break this assumption and are not supported here).
    illuminant_name : str
        Name of the illuminant to use for the integration (e.g., "D65", "A", "F11").
        Must exist in `colour.SDS_ILLUMINANTS`.
    observer_name :  str
        Standard observer to use (e.g., "CIE 1931 2 Degree Standard Observer",
        "CIE 1964 10 Degree Standard Observer"). Must exist in `colour.MSDS_CMFS`.
    interval_nm: float
        Resampling step for the common spectral grid. 1 nm is typical; use ≤ 5 nm for accuracy.

    Returns
    -------
    hex_color : str
        HEX string like '#6096ff'.
    rgb01 : 3D tuple
        Tuple of sRGB components as floats in [0, 1].
    rgb255 : 3D tuple
        Tuple of sRGB components as integers in [0, 255].

    Notes
    -----
    - This computes XYZ by integrating R(λ) * I(λ) * CMF(λ) over the **common** wavelength range
      of the sample, illuminant, and CMFs, then converts XYZ → sRGB with the proper OETF
      (gamma/transfer function).
    - Chromatic adaptation: `colour.XYZ_to_RGB(..., colourspace="sRGB", illuminant=xy_in, 
      chromatic_adaptation_transform="CAT02")` adapts from the input-XYZ illuminant to sRGB's D65.
      If your input illuminant is already D65, this is effectively identity.
    - Do not extrapolate Illuminants or CMFs; wavelengths outside the overlap are ignored.
    """

    # ---------- 0) Validate & prepare inputs --------------------------------
    wls_um = np.asarray(wls_um, dtype=float).ravel()
    R = np.asarray(R, dtype=float).ravel()

    if wls_um.size != R.size:
        raise ValueError("wls_um and R must have the same length.")

    if np.any(~np.isfinite(wls_um)) or np.any(~np.isfinite(R)):
        raise ValueError("wls_um and R must be finite numbers (no NaN/Inf).")

    if interval_nm <= 0:
        raise ValueError("interval_nm must be > 0.")

    # Enforce [0, 1] reflectance range (non-fluorescent assumption).
    R = np.clip(R, 0.0, 1.0)

    # Ensure wavelengths are strictly ascending; sort if needed.
    sort_idx = np.argsort(wls_um)
    wls_um = wls_um[sort_idx]
    R = R[sort_idx]

    # Convert μm → nm for colour-science spectral objects.
    wls_nm = wls_um * 1000.0

    # ---------- 1) Build spectral objects -----------------------------------
    # Sample reflectance as a SpectralDistribution.
    # (If you have duplicate wavelengths, last one wins in dict; avoid duplicates upstream.)
    sd_R = clr.SpectralDistribution(dict(zip(wls_nm, R)), name="sample")

    # Illuminant (spectral power distribution) and CMFs (cone sensitivities).
    try:
        sd_Ill = clr.SDS_ILLUMINANTS[illuminant_name]
    except KeyError as e:
        raise KeyError(f"Unknown illuminant '{illuminant_name}'.") from e

    try:
        cmfs = clr.MSDS_CMFS[observer_name]
    except KeyError as e:
        raise KeyError(f"Unknown observer '{observer_name}'.") from e

    # ---------- 2) Make a common spectral grid (intersection) ---------------
    # We only integrate over wavelengths where *all three* are defined.
    start = max(sd_R.shape.start, sd_Ill.shape.start, cmfs.shape.start)
    end   = min(sd_R.shape.end,   sd_Ill.shape.end,   cmfs.shape.end)
    if not (end > start):
        raise ValueError(
            "No spectral overlap between sample, illuminant, and CMFs "
            f"(sample: {sd_R.shape}, illum: {sd_Ill.shape}, cmfs: {cmfs.shape})."
        )

    shape = clr.SpectralShape(start, end, interval_nm)

    # Interpolate (to fill each nm) and align (ensure grids match exactly)
    sd_R   = sd_R.copy().interpolate(shape).align(shape)
    sd_Ill = sd_Ill.copy().interpolate(shape).align(shape)
    cmfs   = cmfs.copy().interpolate(shape).align(shape)

    # ---------- 3) Spectrum → XYZ under the chosen illuminant/observer ------
    # sd_to_XYZ returns Y=100 for a perfect diffuser under the given illuminant.
    XYZ = clr.sd_to_XYZ(sd_R, cmfs=cmfs, illuminant=sd_Ill)  # shape (3,)
    # Normalise to 0..1 range for RGB conversion (Colour expects XYZ relative to 1.0)
    XYZ_n = XYZ / 100.0

    # ---------- 4) XYZ (input illuminant) → sRGB (D65) ----------------------
    # sRGB colourspace object (contains whitepoint, matrices, transfer functions).
    srgb = clr.RGB_COLOURSPACES["sRGB"]

    # xy chromaticity *of the input XYZ* (i.e., the illuminant used for integration).
    # This tells Colour what to adapt *from*; it adapts to sRGB's white (D65) by default.
    xy_in = clr.CCS_ILLUMINANTS[observer_name][illuminant_name]

    rgb = clr.XYZ_to_RGB(
        XYZ_n,
        srgb,                                  # Target colourspace
        illuminant=xy_in,                      # Input XYZ illuminant (xy)
        chromatic_adaptation_transform="CAT02",# Adapt from input → D65
        apply_cctf_encoding=True,              # Apply sRGB OETF (gamma)
    )

    # Keep inside displayable range; out-of-gamut values can occur.
    rgb = np.clip(rgb, 0.0, 1.0)

    # ---------- 5) Pack outputs ---------------------------------------------
    rgb255 = (rgb * 255.0 + 0.5).astype(int)
    hex_color = f"#{rgb255[0]:02x}{rgb255[1]:02x}{rgb255[2]:02x}"
    return hex_color, (float(rgb[0]), float(rgb[1]), float(rgb[2])), (int(rgb255[0]), int(rgb255[1]), int(rgb255[2]))

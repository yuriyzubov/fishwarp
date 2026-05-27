"""
Diagnostic for the stage 4 'images do not sufficiently overlap' error.

Reads the blurred common-grid images and the saved COM transform, then checks
whether the COM translation actually brings the two foregrounds together.
Run from the registration directory.
"""

import ants
import numpy as np

import config
from utils import load_volume, to_ants


def _stats(name, arr):
    fg = arr > 0
    print(f'[{name}]  shape={arr.shape}  dtype={arr.dtype}')
    print(f'    min={arr.min():.6f}  max={arr.max():.6f}  '
          f'mean={arr.mean():.6f}  sum={arr.sum():.3f}')
    print(f'    foreground (>0): {int(fg.sum())} / {arr.size} voxels')
    if fg.any():
        idx = np.argwhere(fg)
        lo = idx.min(0)
        hi = idx.max(0)
        print(f'    fg bbox (vox, z,y,x): lo={tuple(int(v) for v in lo)}  '
              f'hi={tuple(int(v) for v in hi)}')
        return lo, hi
    print('    *** EMPTY — no foreground voxels ***')
    return None, None


def main():
    spacing_mm = config.COMMON_GRID_SPACING_UM / 1000.0
    print(f'Common grid spacing: {spacing_mm} mm\n')

    fixed  = load_volume(config.INTERMEDIATE / 'fixed_blur.zarr')
    moving = load_volume(config.INTERMEDIATE / 'moving_blur.zarr')

    print('=== blur image stats ===')
    f_lo, f_hi = _stats('fixed_blur',  fixed)
    m_lo, m_hi = _stats('moving_blur', moving)
    print()

    # --- COM via ANTs, exactly as stage 3 computes it --------------------
    iso_nm = (config.COMMON_GRID_SPACING_UM * 1000,) * 3
    fixed_com  = np.array(ants.get_center_of_mass(to_ants(fixed,  iso_nm)))
    moving_com = np.array(ants.get_center_of_mass(to_ants(moving, iso_nm)))
    translation = moving_com - fixed_com
    print('=== COM (ANTs physical coords, mm) ===')
    print(f'    fixed_com   = {fixed_com}')
    print(f'    moving_com  = {moving_com}')
    print(f'    translation = moving - fixed = {translation}')
    if np.any(~np.isfinite(fixed_com)) or np.any(~np.isfinite(moving_com)):
        print('    *** COM contains nan/inf — degenerate image ***')
    print()

    # --- Saved transform --------------------------------------------------
    tx = ants.read_transform(str(config.TRANSFORMS / 'com_translation.mat'))
    params = np.array(tx.parameters)
    print('=== com_translation.mat ===')
    print(f'    parameters        = {params}')
    print(f'    -> translation    = {params[9:12]}')
    print(f'    fixed_parameters  = {np.array(tx.fixed_parameters)}')
    print()

    # --- Overlap check ----------------------------------------------------
    # apply_transforms is pull-based: output(x) samples moving at x + t, so in
    # the fixed/output frame the moving foreground sits at [m_lo - t, m_hi - t].
    if f_lo is None or m_lo is None:
        print('Cannot check overlap — a blur image is empty (see above).')
        return

    # bbox in mm, numpy (z,y,x) order
    f_lo_mm, f_hi_mm = f_lo * spacing_mm, f_hi * spacing_mm
    m_lo_mm, m_hi_mm = m_lo * spacing_mm, m_hi * spacing_mm
    print('=== foreground bbox (mm, z,y,x) ===')
    print(f'    fixed : lo={np.round(f_lo_mm,3)}  hi={np.round(f_hi_mm,3)}')
    print(f'    moving: lo={np.round(m_lo_mm,3)}  hi={np.round(m_hi_mm,3)}')

    # get_center_of_mass returns ANTs dim order == numpy axis order here.
    t = translation
    m_lo_shifted = m_lo_mm - t
    m_hi_shifted = m_hi_mm - t
    print(f'    moving after COM shift: lo={np.round(m_lo_shifted,3)}  '
          f'hi={np.round(m_hi_shifted,3)}')

    inter_lo = np.maximum(f_lo_mm, m_lo_shifted)
    inter_hi = np.minimum(f_hi_mm, m_hi_shifted)
    overlap = np.all(inter_hi > inter_lo)
    print(f'\n    bboxes overlap after COM init? {overlap}')
    if overlap:
        ovl = np.prod(inter_hi - inter_lo)
        fvol = np.prod(f_hi_mm - f_lo_mm)
        print(f'    overlap volume / fixed bbox volume = {ovl / fvol:.3f}')
    else:
        gap = inter_lo - inter_hi
        print(f'    gap per axis (mm): {np.round(gap, 3)}  '
              '(positive = no overlap on that axis)')


if __name__ == '__main__':
    main()

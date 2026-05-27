"""
Stage 1: Compute signed distance transforms (SDF) for fixed and moving binary masks.

SDF convention: positive outside nuclei, negative inside.
Anisotropic voxel spacings are passed to edt so distances are in physical units (mm).
Output is clipped and saved as float32.

Outputs:
  outputs/intermediate/fixed_sdf.zarr
  outputs/intermediate/moving_sdf.zarr
"""

import logging

import edt
import numpy as np

import config
from utils import load_volume, save_volume, timed

log = logging.getLogger(__name__)


def compute_sdf(mask: np.ndarray, spacing_nm: tuple, name: str) -> np.ndarray:
    """
    Compute signed distance transform for a binary mask.
    spacing_nm is a (z, y, x) tuple in nanometers — passed directly to edt as anisotropy.
    Returns float32 SDF clipped to ±SDF_CLIP_VOXELS voxel-widths (using minimum spacing).
    """
    spacing_mm = tuple(s / 1e6 for s in spacing_nm)  # (z, y, x) in mm, matches array axes
    min_spacing_mm = min(spacing_mm)
    clip_mm = config.SDF_CLIP_VOXELS * min_spacing_mm

    log.info('%s: spacing_mm=%s  clip=±%.4f mm (±%d voxels × %.4f mm)',
             name, spacing_mm, clip_mm, config.SDF_CLIP_VOXELS, min_spacing_mm)

    with timed(f'{name} EDT inside'):
        inside = edt.edt(mask, anisotropy=spacing_mm, parallel=config.EDT_PARALLEL_JOBS)

    with timed(f'{name} EDT outside'):
        outside = edt.edt(1 - mask, anisotropy=spacing_mm, parallel=config.EDT_PARALLEL_JOBS)

    with timed(f'{name} SDF assembly + clip'):
        sdf = (outside - inside).astype(np.float32)
        sdf = np.clip(sdf, -clip_mm, clip_mm)

    log.info('%s SDF stats: min=%.4f  max=%.4f  mean=%.4f  std=%.4f',
             name, sdf.min(), sdf.max(), sdf.mean(), sdf.std())

    return sdf


def run():
    fixed_binary  = load_volume(config.INTERMEDIATE / 'fixed_binary.zarr')
    moving_binary = load_volume(config.INTERMEDIATE / 'moving_binary.zarr')

    assert fixed_binary.ndim == 3,  f'Expected 3D fixed mask, got shape {fixed_binary.shape}'
    assert moving_binary.ndim == 3, f'Expected 3D moving mask, got shape {moving_binary.shape}'

    with timed('Stage 1 fixed SDF'):
        fixed_sdf = compute_sdf(fixed_binary, config.FIXED_SPACING_NM, 'fixed')

    with timed('Stage 1 moving SDF'):
        moving_sdf = compute_sdf(moving_binary, config.MOVING_SPACING_NM, 'moving')

    save_volume(config.INTERMEDIATE / 'fixed_sdf.zarr',  fixed_sdf)
    save_volume(config.INTERMEDIATE / 'moving_sdf.zarr', moving_sdf)

    log.info('Stage 1 complete.')


if __name__ == '__main__':
    run()

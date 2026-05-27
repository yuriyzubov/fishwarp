"""
Stage 1b: Build registration images on a common isotropic grid.

Resamples both binary masks to a shared grid (origin (0,0,0), isotropic spacing
COMMON_GRID_SPACING_UM, shape large enough to contain both physical extents),
then Gaussian-blurs at sigma = BLUR_SIGMA_UM.

The resample step is chunked (utils.resample_to_grid_chunked): the output grid
is tiled and each worker reads only the small source slab it needs — the
full-resolution binary is never held in RAM. The blur runs once on the whole
common-grid array, which is small (~1000× fewer voxels than full res), so
there is no memory reason to chunk it.

Outputs (used when config.REG_IMAGE_TYPE == 'blur'):
  intermediate/fixed_blur.zarr
  intermediate/moving_blur.zarr
"""

import logging

import ants
import numpy as np
import zarr

import config
from utils import resample_to_grid_chunked, save_volume, timed

log = logging.getLogger(__name__)


def _common_grid_shape(fixed_shape, fixed_spacing_nm,
                       moving_shape, moving_spacing_nm,
                       target_spacing_mm):
    """
    Target shape for a common isotropic grid that fully contains both volumes'
    physical extents. Returned in numpy axis order (same as the source arrays).

    ants.resample_image_to_target matches numpy axes positionally (axis 0 ↔
    axis 0, …), so the target shape MUST stay in source axis order — an
    x/y/z-reordered shape sends the large axis into the small one and crops.

    numpy axis a carries spacing_nm[a] (see to_ants); the physical extent of a
    volume along numpy axis a is therefore shape[a] * spacing_nm[a].
    """
    def extent_mm(shape, spacing_nm):
        return tuple(s * p / 1e6 for s, p in zip(shape, spacing_nm))

    fixed_ext  = extent_mm(fixed_shape,  fixed_spacing_nm)
    moving_ext = extent_mm(moving_shape, moving_spacing_nm)
    max_ext    = tuple(max(f, m) for f, m in zip(fixed_ext, moving_ext))
    target_shape = tuple(int(np.ceil(e / target_spacing_mm)) for e in max_ext)
    return target_shape, max_ext


def _resample_and_blur(src_zarr, native_spacing_nm_zyx, target_shape,
                       target_spacing_mm, sigma_mm, name, n_workers):
    """
    Chunked-resample a binary zarr onto the common grid, then Gaussian-blur the
    whole (small) common-grid array. Returns numpy array, (z, y, x), float32.
    """
    # Per-numpy-axis spacing in mm — same (z, y, x) order as the array, matching
    # to_ants/ants.from_numpy (which applies spacing[k] to numpy axis k).
    spacing_mm = tuple(s / 1e6 for s in native_spacing_nm_zyx)
    log.info('%s native: shape=%s  spacing_mm=%s',
             name, tuple(src_zarr.shape), spacing_mm)

    with timed(f'{name} chunked resample → common grid'):
        resampled = resample_to_grid_chunked(
            src_zarr, spacing_mm, target_shape, target_spacing_mm,
            name=name, n_workers=n_workers,
        )

    res_ants = ants.from_numpy(resampled, origin=(0.0, 0.0, 0.0),
                               spacing=(target_spacing_mm,) * 3)
    with timed(f'{name} Gaussian blur (sigma={sigma_mm} mm)'):
        blurred = ants.smooth_image(
            res_ants, sigma=sigma_mm, sigma_in_physical_coordinates=True,
        )

    arr = blurred.numpy().astype(np.float32)
    log.info('%s blurred stats: min=%.4f max=%.4f mean=%.4f',
             name, arr.min(), arr.max(), arr.mean())
    return arr


def run():
    config.INTERMEDIATE.mkdir(parents=True, exist_ok=True)

    n_workers = config.STAGE1B_N_WORKERS

    # Open binary zarrs lazily — resample reads only small slabs per tile.
    fixed_zarr  = zarr.open_array(str(config.INTERMEDIATE / 'fixed_binary.zarr'),  mode='r')
    moving_zarr = zarr.open_array(str(config.INTERMEDIATE / 'moving_binary.zarr'), mode='r')

    target_spacing_mm = config.COMMON_GRID_SPACING_UM / 1000.0
    sigma_mm          = config.BLUR_SIGMA_UM         / 1000.0

    target_shape, max_extent_mm = _common_grid_shape(
        fixed_zarr.shape,  config.FIXED_SPACING_NM,
        moving_zarr.shape, config.MOVING_SPACING_NM,
        target_spacing_mm,
    )
    log.info('Common grid: spacing=%.4f mm  shape(z,y,x)=%s  extent_mm=%s',
             target_spacing_mm, target_shape,
             tuple(round(e, 4) for e in max_extent_mm))

    fixed_arr  = _resample_and_blur(fixed_zarr,  config.FIXED_SPACING_NM,
                                    target_shape, target_spacing_mm,
                                    sigma_mm, 'fixed', n_workers)
    moving_arr = _resample_and_blur(moving_zarr, config.MOVING_SPACING_NM,
                                    target_shape, target_spacing_mm,
                                    sigma_mm, 'moving', n_workers)

    save_volume(config.INTERMEDIATE / 'fixed_blur.zarr',  fixed_arr)
    save_volume(config.INTERMEDIATE / 'moving_blur.zarr', moving_arr)

    log.info('Stage 1b complete.')


if __name__ == '__main__':
    run()

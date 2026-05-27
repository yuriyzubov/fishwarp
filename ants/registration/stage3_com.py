"""
Stage 3: Center-of-mass translation.

Pure numpy — no optimizer, completes in seconds.
Computes the translation that aligns the centers of mass of the two
registration images and saves it as an ANTs AffineTransform (identity
rotation, COM offset). In 'blur' mode the COM is taken from the same
common-grid blurred images stages 4/5 register on, so the init lives in
the identical physical frame as the downstream optimization.

Outputs:
  transforms/com_translation.mat
  intermediate/moving_binary_warped_com.zarr   (only if warp=True)
"""

import logging

import ants
import click
import numpy as np

import config
from utils import (load_reg_images, load_volume, quick_dice, save_volume,
                   timed, to_ants, warp_binary)

log = logging.getLogger(__name__)


def _foreground_bbox_mm(binary: np.ndarray, spacing_nm: tuple) -> dict:
    """Foreground bounding box in (z,y,x) physical mm, plus extent and voxel counts."""
    coords = np.argwhere(binary > 0)
    if coords.size == 0:
        return {'empty': True}
    lo_vox = coords.min(axis=0)
    hi_vox = coords.max(axis=0) + 1  # exclusive
    spacing_mm = np.array(spacing_nm) / 1e6
    lo_mm = lo_vox * spacing_mm
    hi_mm = hi_vox * spacing_mm
    return {
        'shape_vox':   tuple(int(s) for s in binary.shape),
        'spacing_mm':  tuple(float(s) for s in spacing_mm),
        'bbox_lo_vox': tuple(int(v) for v in lo_vox),
        'bbox_hi_vox': tuple(int(v) for v in hi_vox),
        'bbox_lo_mm':  tuple(round(float(v), 4) for v in lo_mm),
        'bbox_hi_mm':  tuple(round(float(v), 4) for v in hi_mm),
        'extent_mm':   tuple(round(float(v), 4) for v in (hi_mm - lo_mm)),
        'fg_voxels':   int(coords.shape[0]),
    }


def run(warp: bool = False):
    config.TRANSFORMS.mkdir(parents=True, exist_ok=True)
    config.QC.mkdir(parents=True, exist_ok=True)

    fixed_binary  = load_volume(config.INTERMEDIATE / 'fixed_binary.zarr')
    moving_binary = load_volume(config.INTERMEDIATE / 'moving_binary.zarr')

    with timed('foreground bbox diagnostics'):
        fixed_bbox  = _foreground_bbox_mm(fixed_binary,  config.FIXED_SPACING_NM)
        moving_bbox = _foreground_bbox_mm(moving_binary, config.MOVING_SPACING_NM)

    extents_log = config.QC / 'stage3_extents.log'
    lines = ['# Stage 3 foreground extent diagnostics (axis order: z, y, x)\n']
    for name, bbox in (('fixed', fixed_bbox), ('moving', moving_bbox)):
        lines.append(f'\n[{name}]\n')
        for k, v in bbox.items():
            lines.append(f'  {k:12s} : {v}\n')
    if not fixed_bbox.get('empty') and not moving_bbox.get('empty'):
        ratio = tuple(round(m / f, 3) for m, f in
                      zip(moving_bbox['extent_mm'], fixed_bbox['extent_mm']))
        lines.append(f'\n[ratio moving/fixed extent_mm] : {ratio}\n')
    extents_log.write_text(''.join(lines))
    log.info('Wrote foreground extent diagnostics → %s', extents_log)
    log.info('Fixed  fg extent (z,y,x) mm: %s', fixed_bbox.get('extent_mm'))
    log.info('Moving fg extent (z,y,x) mm: %s', moving_bbox.get('extent_mm'))

    with timed('COM computation'):
        # Compute the COM from the SAME images stages 4/5 register on, so the
        # COM init lives in the identical physical frame as the downstream
        # rigid/affine optimization. In 'blur' mode that is the common-grid
        # blurred image; in 'sdf' mode COM on a signed distance field is
        # meaningless, so fall back to the native binary masks.
        if config.REG_IMAGE_TYPE == 'blur':
            fixed_ants, moving_ants = load_reg_images()
            log.info('COM source: blurred common-grid images')
        else:
            fixed_ants  = to_ants(fixed_binary.astype(np.float32),  config.FIXED_SPACING_NM)
            moving_ants = to_ants(moving_binary.astype(np.float32), config.MOVING_SPACING_NM)
            log.info('COM source: native binary masks')

        fixed_com  = np.array(ants.get_center_of_mass(fixed_ants))   # (x,y,z) mm
        moving_com = np.array(ants.get_center_of_mass(moving_ants))  # (x,y,z) mm
        # ants.apply_transforms is pull-based (fixed → moving), so the transform
        # must map fixed points to moving points. To align COMs, a fixed voxel at
        # fixed_com must sample from moving_com → translation = moving_com - fixed_com.
        translation_xyz = moving_com - fixed_com

    log.info('Fixed  COM (x,y,z) mm : %s', np.round(fixed_com,       4))
    log.info('Moving COM (x,y,z) mm : %s', np.round(moving_com,      4))
    log.info('Translation (x,y,z) mm: %s', np.round(translation_xyz, 4))

    tx = ants.create_ants_transform(
        transform_type='AffineTransform', precision='float', dimension=3,
    )
    tx.set_parameters(np.concatenate([np.eye(3).flatten(), translation_xyz]))
    # Rotation center at fixed COM (not origin). With identity rotation this
    # has no effect on alignment, but when this transform is composed with the
    # rigid/affine that follow, a rotation pivot near the data avoids the
    # lever-arm pathology of pivoting around the volume corner.
    tx.set_fixed_parameters(fixed_com.tolist())

    com_tx_path = str(config.TRANSFORMS / 'com_translation.mat')
    ants.write_transform(tx, com_tx_path)
    log.info('COM transform saved: %s', com_tx_path)

    if warp:
        with timed('apply COM to moving binary'):
            warped = warp_binary(
                moving_binary, config.MOVING_SPACING_NM,
                fixed_binary,  config.FIXED_SPACING_NM,
                [com_tx_path],
            )

        log.info('Post-COM Dice: %.4f', quick_dice(fixed_binary, warped))
        save_volume(config.INTERMEDIATE / 'moving_binary_warped_com.zarr', warped)
    else:
        log.info('Skipping warp/QC (pass --warp to apply transform and save warped zarr).')

    log.info('Stage 3 complete.')


@click.command()
@click.option('--warp', is_flag=True, default=False,
              help='Apply COM transform to moving binary and save warped zarr for QC.')
def main(warp):
    run(warp=warp)


if __name__ == '__main__':
    main()

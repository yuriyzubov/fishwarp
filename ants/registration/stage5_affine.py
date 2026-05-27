"""
Stage 5: Affine registration initialized from rigid chain.

Loads registration images via utils.load_reg_images (dispatches on
config.REG_IMAGE_TYPE). Uses ANTs' default Mattes MI metric.

Outputs:
  transforms/affine_0GenericAffine.mat
  transforms/transform_manifest.json
  intermediate/moving_binary_warped_affine.zarr
  intermediate/moving_sdf_warped_affine.zarr   (only when REG_IMAGE_TYPE='sdf')
"""

import json
import logging

import ants
import click
import numpy as np

import config
from utils import (from_ants, load_reg_images, load_volume, quick_dice,
                   save_volume, timed, to_ants, warp_binary)

log = logging.getLogger(__name__)


def run(warp: bool = False):
    config.TRANSFORMS.mkdir(parents=True, exist_ok=True)

    for name in ('com_translation.mat', 'rigid.mat', 'rigid_inv.mat'):
        assert (config.TRANSFORMS / name).exists(), \
            f'Missing {name} — run stage3_com and stage4_rigid first'

    # rigid.mat is self-contained — ants.registration collapsed the COM init
    # into it. Use it alone to initialize affine; re-chaining
    # com_translation.mat would double-count the COM translation.
    rigid_tx = str(config.TRANSFORMS / 'rigid.mat')

    fixed_ds, moving_ds = load_reg_images(config.AFFINE_DOWNSAMPLE_FACTOR)

    log.info('Affine registration (REG_IMAGE_TYPE=%s, initialized from rigid.mat) …',
             config.REG_IMAGE_TYPE)
    with timed('Affine registration'):
        affine_result = ants.registration(
            fixed=fixed_ds,
            moving=moving_ds,
            type_of_transform='Affine',
            initial_transform=[rigid_tx],
            outprefix=str(config.TRANSFORMS / 'affine_'),
            verbose=True,
        )

    affine_tx = affine_result['fwdtransforms'][0]
    log.info('Affine transform: %s', affine_tx)

    # affine.mat is self-contained: ants.registration collapsed the rigid
    # init (which itself already contains COM) into it. The full forward
    # transform is therefore this single GenericAffine — no manual chaining.
    full_fwd = [affine_tx]
    log.info('Full forward transform: %s', full_fwd)

    manifest = {
        'fwdtransforms': full_fwd,
        'invtransforms': [affine_result['invtransforms'][0]],
    }
    (config.TRANSFORMS / 'transform_manifest.json').write_text(
        json.dumps(manifest, indent=2)
    )
    log.info('Transform manifest saved.')

    if warp:
        # -- Apply to binary mask (fast, for immediate QC) -------------------
        fixed_binary  = load_volume(config.INTERMEDIATE / 'fixed_binary.zarr')
        moving_binary = load_volume(config.INTERMEDIATE / 'moving_binary.zarr')

        with timed('apply full chain to moving binary'):
            warped_binary = warp_binary(
                moving_binary, config.MOVING_SPACING_NM,
                fixed_binary,  config.FIXED_SPACING_NM,
                full_fwd,
            )

        log.info('Post-affine Dice: %.4f', quick_dice(fixed_binary, warped_binary))
        save_volume(config.INTERMEDIATE / 'moving_binary_warped_affine.zarr', warped_binary)

        # -- Apply to full-res SDF (only when SDFs were used / are available) --
        if config.REG_IMAGE_TYPE == 'sdf':
            fixed_sdf_path = (
                config.INTERMEDIATE / 'fixed_sdf_blurred.zarr'
                if config.ENABLE_STAGE2_BLUR
                else config.INTERMEDIATE / 'fixed_sdf.zarr'
            )
            fixed_sdf  = load_volume(fixed_sdf_path)
            moving_sdf = load_volume(config.INTERMEDIATE / 'moving_sdf.zarr')

            fixed_full  = to_ants(fixed_sdf,  config.FIXED_SPACING_NM)
            moving_full = to_ants(moving_sdf, config.MOVING_SPACING_NM)

            with timed('apply full chain to full-res moving SDF'):
                warped_sdf = from_ants(ants.apply_transforms(
                    fixed=fixed_full,
                    moving=moving_full,
                    transformlist=full_fwd,
                    interpolator='linear',
                )).astype(np.float32)

            log.info('Warped SDF stats: min=%.4f  max=%.4f  mean=%.4f',
                     warped_sdf.min(), warped_sdf.max(), warped_sdf.mean())
            save_volume(config.INTERMEDIATE / 'moving_sdf_warped_affine.zarr', warped_sdf)
        else:
            log.info('Skipping warped-SDF output (REG_IMAGE_TYPE=%s, SDFs not built).',
                     config.REG_IMAGE_TYPE)
    else:
        log.info('Skipping warp/QC outputs (pass --warp to apply chain to binary and SDF).')

    log.info('Stage 5 complete.')


@click.command()
@click.option('--warp', is_flag=True, default=False,
              help='Apply full chain to moving binary and SDF; save warped zarrs for QC and stage 6.')
def main(warp):
    run(warp=warp)


if __name__ == '__main__':
    main()

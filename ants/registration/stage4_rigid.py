"""
Stage 4: Rigid registration initialized from COM translation.

Loads registration images via utils.load_reg_images, which dispatches on
config.REG_IMAGE_TYPE ('sdf' or 'blur'). The default ANTs metric (Mattes MI)
is used.

Outputs:
  transforms/rigid.mat
  transforms/rigid_inv.mat
  intermediate/moving_binary_warped_rigid.zarr
"""

import logging
import shutil

import ants
import click
import numpy as np

import config
from utils import (load_reg_images, load_volume, quick_dice,
                   save_volume, timed, warp_binary)

log = logging.getLogger(__name__)


def run(warp: bool = False):
    config.TRANSFORMS.mkdir(parents=True, exist_ok=True)

    com_tx_path = str(config.TRANSFORMS / 'com_translation.mat')
    assert (config.TRANSFORMS / 'com_translation.mat').exists(), \
        'Missing com_translation.mat — run stage3_com first'

    fixed_ds, moving_ds = load_reg_images(config.AFFINE_DOWNSAMPLE_FACTOR)

    log.info('Rigid registration (REG_IMAGE_TYPE=%s, initialized from COM) …',
             config.REG_IMAGE_TYPE)
    with timed('Rigid registration'):
        rigid_result = ants.registration(
            fixed=fixed_ds,
            moving=moving_ds,
            type_of_transform='Rigid',
            initial_transform=com_tx_path,
            outprefix=str(config.TRANSFORMS / 'rigid_'),
            verbose=True,
        )

    rigid_tx = str(config.TRANSFORMS / 'rigid.mat')
    shutil.copy(rigid_result['fwdtransforms'][0], rigid_tx)
    rigid_inv_tx = str(config.TRANSFORMS / 'rigid_inv.mat')
    shutil.copy(rigid_result['invtransforms'][0], rigid_inv_tx)
    log.info('Rigid transforms saved: %s', rigid_tx)

    # rigid.mat is self-contained: ants.registration collapses the initial
    # (COM) transform into its output GenericAffine. Re-chaining
    # com_translation.mat here would apply the COM translation twice.
    rigid_chain = [rigid_tx]

    if warp:
        fixed_binary  = load_volume(config.INTERMEDIATE / 'fixed_binary.zarr')
        moving_binary = load_volume(config.INTERMEDIATE / 'moving_binary.zarr')

        with timed('apply rigid chain to moving binary'):
            warped = warp_binary(
                moving_binary, config.MOVING_SPACING_NM,
                fixed_binary,  config.FIXED_SPACING_NM,
                rigid_chain,
            )

        log.info('Post-rigid Dice: %.4f', quick_dice(fixed_binary, warped))
        save_volume(config.INTERMEDIATE / 'moving_binary_warped_rigid.zarr', warped)
    else:
        log.info('Skipping warp/QC (pass --warp to apply transform and save warped zarr).')

    log.info('Stage 4 complete.')


@click.command()
@click.option('--warp', is_flag=True, default=False,
              help='Apply rigid chain to moving binary and save warped zarr for QC.')
def main(warp):
    run(warp=warp)


if __name__ == '__main__':
    main()

"""
Stage 5b: SyN deformable registration, initialized from the affine.

Adds a smooth nonlinear warp on top of the affine to correct local shape
differences a global affine cannot represent — e.g. a curved body axis, where
red (fixed) and green (warped) diverge toward the tail in the stage 6 slices.

Registers the same images as stages 4/5 (utils.load_reg_images) and starts
from stage 5's affine transform. Tunables live in config (SYN_FLOW_SIGMA,
SYN_TOTAL_SIGMA, SYN_ITERATIONS).

Outputs:
  transforms/syn_1Warp.nii.gz
  transforms/syn_1InverseWarp.nii.gz
  transforms/transform_manifest.json   (overwritten — fwd/inv now include SyN)
  intermediate/moving_binary_warped_syn.zarr
"""

import json
import logging

import ants
import click

import config
from utils import (load_reg_images, load_volume, quick_dice,
                   save_volume, timed, warp_binary)

log = logging.getLogger(__name__)


def run(warp: bool = False):
    config.TRANSFORMS.mkdir(parents=True, exist_ok=True)

    affine_tx = str(config.TRANSFORMS / 'affine_0GenericAffine.mat')
    assert (config.TRANSFORMS / 'affine_0GenericAffine.mat').exists(), \
        'Missing affine_0GenericAffine.mat — run stage5_affine first'

    fixed, moving = load_reg_images(config.SYN_DOWNSAMPLE_FACTOR)

    log.info('SyN deformable registration (REG_IMAGE_TYPE=%s, initialized from affine) …',
             config.REG_IMAGE_TYPE)
    log.info('  flow_sigma=%s  total_sigma=%s  reg_iterations=%s',
             config.SYN_FLOW_SIGMA, config.SYN_TOTAL_SIGMA, config.SYN_ITERATIONS)
    with timed('SyN registration'):
        syn_result = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform='SyNOnly',
            initial_transform=[affine_tx],
            flow_sigma=config.SYN_FLOW_SIGMA,
            total_sigma=config.SYN_TOTAL_SIGMA,
            reg_iterations=config.SYN_ITERATIONS,
            outprefix=str(config.TRANSFORMS / 'syn_'),
            verbose=True,
        )

    # ants.registration returns the COMPLETE forward/inverse transform lists
    # (warp field + the affine init it was started from). Use them verbatim —
    # never re-chain the initial transform by hand. Hand-chaining is exactly
    # what double-counted the COM translation in an earlier version of the
    # rigid/affine stages.
    full_fwd = list(syn_result['fwdtransforms'])
    full_inv = list(syn_result['invtransforms'])
    log.info('Full forward transform list:  %s', full_fwd)
    log.info('Full inverse transform list:  %s', full_inv)

    manifest = {
        'fwdtransforms': full_fwd,
        'invtransforms': full_inv,
    }
    (config.TRANSFORMS / 'transform_manifest.json').write_text(
        json.dumps(manifest, indent=2)
    )
    log.info('Transform manifest saved (now includes the SyN warp).')

    if warp:
        fixed_binary  = load_volume(config.INTERMEDIATE / 'fixed_binary.zarr')
        moving_binary = load_volume(config.INTERMEDIATE / 'moving_binary.zarr')

        with timed('apply full chain to moving binary'):
            warped_binary = warp_binary(
                moving_binary, config.MOVING_SPACING_NM,
                fixed_binary,  config.FIXED_SPACING_NM,
                full_fwd,
            )

        log.info('Post-SyN Dice: %.4f', quick_dice(fixed_binary, warped_binary))
        save_volume(config.INTERMEDIATE / 'moving_binary_warped_syn.zarr', warped_binary)
    else:
        log.info('Skipping warp/QC (pass --warp to apply chain and save warped zarr).')

    log.info('Stage 5b complete.')


@click.command()
@click.option('--warp', is_flag=True, default=False,
              help='Apply full chain to moving binary and save warped zarr for QC.')
def main(warp):
    run(warp=warp)


if __name__ == '__main__':
    main()

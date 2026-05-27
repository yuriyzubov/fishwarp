"""
Stage 5c: Refine deformable registration with another SyN pass on top of the
existing transform chain.

The careful pattern:
  1. Read the current transform_manifest.json (whatever combination of
     affine + SyN it currently encodes).
  2. Pre-warp moving_blur with that chain — the result is an image roughly
     aligned to fixed_blur, on the fixed (common) grid.
  3. Run a FRESH SyN between fixed_blur and the pre-warped moving — no
     initial_transform, so the new fwdtransforms is purely the residual
     warp. No ambiguity about what ANTsPy folds or doesn't.
  4. Compose explicitly. The new warp lives on the fixed grid, so in the
     pull-based right-to-left convention it goes at the END of fwdtransforms
     (it acts on a fixed point BEFORE prev_chain pulls it back to moving).
     Its inverse goes at the BEGINNING of invtransforms.

Re-running stage5c picks up the manifest produced by the previous run, so
multiple refinements stack cleanly.

Outputs:
  transforms/syn_refineN_1Warp.nii.gz
  transforms/syn_refineN_1InverseWarp.nii.gz   (N = next unused index)
  transforms/transform_manifest.json           (overwritten with composed chain)
  intermediate/moving_binary_warped_syn_refine.zarr   (only with --warp)
"""

import json
import logging
import re

import ants
import click

import config
from utils import (load_reg_images, load_volume, quick_dice,
                   save_volume, timed, warp_binary)

log = logging.getLogger(__name__)


def _next_refine_index():
    pattern = re.compile(r'syn_refine(\d+)_1Warp\.nii\.gz$')
    existing = [int(m.group(1))
                for p in config.TRANSFORMS.glob('syn_refine*_1Warp.nii.gz')
                for m in [pattern.fullmatch(p.name)] if m]
    return max(existing) + 1 if existing else 1


def run(warp: bool = False):
    config.TRANSFORMS.mkdir(parents=True, exist_ok=True)

    manifest_path = config.TRANSFORMS / 'transform_manifest.json'
    assert manifest_path.exists(), (
        'Missing transform_manifest.json — run stage5_affine (or stage5b_syn) first'
    )
    manifest = json.loads(manifest_path.read_text())
    prev_fwd = list(manifest['fwdtransforms'])
    prev_inv = list(manifest['invtransforms'])
    log.info('Previous fwd chain (%d entries): %s', len(prev_fwd), prev_fwd)
    log.info('Previous inv chain (%d entries): %s', len(prev_inv), prev_inv)

    fixed, moving = load_reg_images(config.SYN_DOWNSAMPLE_FACTOR)

    # 1. Pre-warp moving with the existing chain → moving on the fixed grid.
    with timed('pre-warp moving with current chain'):
        moving_prewarped = ants.apply_transforms(
            fixed=fixed, moving=moving,
            transformlist=prev_fwd, interpolator='linear',
        )

    # 2. Fresh SyN on (fixed, moving_prewarped) — no initial_transform.
    idx = _next_refine_index()
    outprefix = str(config.TRANSFORMS / f'syn_refine{idx}_')
    log.info('SyN refinement #%d (fresh, no initial_transform) — '
             'flow_sigma=%s total_sigma=%s reg_iterations=%s',
             idx, config.SYN_FLOW_SIGMA, config.SYN_TOTAL_SIGMA,
             config.SYN_ITERATIONS)
    with timed(f'SyN refine #{idx}'):
        syn_result = ants.registration(
            fixed=fixed,
            moving=moving_prewarped,
            type_of_transform='SyNOnly',
            flow_sigma=config.SYN_FLOW_SIGMA,
            total_sigma=config.SYN_TOTAL_SIGMA,
            reg_iterations=config.SYN_ITERATIONS,
            outprefix=outprefix,
            verbose=True,
        )
    new_fwd = list(syn_result['fwdtransforms'])
    new_inv = list(syn_result['invtransforms'])
    log.info('New SyN fwd: %s', new_fwd)
    log.info('New SyN inv: %s', new_inv)

    # 3. Explicit composition.
    # The new warp acts on the fixed grid: it must rectify a fixed point
    # BEFORE prev_chain pulls that point back into the original moving's
    # grid. apply_transforms applies the list right-to-left, so:
    #   - fwd: new warp goes at the END (innermost / applied first in pull)
    #   - inv: new inverse goes at the BEGINNING (mirror order)
    full_fwd = prev_fwd + new_fwd
    full_inv = new_inv + prev_inv
    log.info('Composed fwd chain (%d entries): %s', len(full_fwd), full_fwd)
    log.info('Composed inv chain (%d entries): %s', len(full_inv), full_inv)

    manifest_path.write_text(
        json.dumps({'fwdtransforms': full_fwd,
                    'invtransforms': full_inv}, indent=2)
    )
    log.info('Transform manifest overwritten with the refined chain.')

    if warp:
        fixed_binary  = load_volume(config.INTERMEDIATE / 'fixed_binary.zarr')
        moving_binary = load_volume(config.INTERMEDIATE / 'moving_binary.zarr')
        with timed('apply refined chain to moving binary'):
            warped_binary = warp_binary(
                moving_binary, config.MOVING_SPACING_NM,
                fixed_binary,  config.FIXED_SPACING_NM,
                full_fwd,
            )
        log.info('Post-refine#%d Dice: %.4f', idx,
                 quick_dice(fixed_binary, warped_binary))
        save_volume(config.INTERMEDIATE / 'moving_binary_warped_syn_refine.zarr',
                    warped_binary)
    else:
        log.info('Skipping warp/QC (pass --warp to apply refined chain and save warped zarr).')

    log.info('Stage 5c complete (refinement #%d).', idx)


@click.command()
@click.option('--warp', is_flag=True, default=False,
              help='Apply refined chain to moving binary and save warped zarr for QC.')
def main(warp):
    run(warp=warp)


if __name__ == '__main__':
    main()

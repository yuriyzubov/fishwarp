"""
Stage 7: Apply the registration transform chain to non-binary moving volumes.

Reads the forward chain from transform_manifest.json (written by stage 5 or
stage 5b — works with either) and warps two moving volumes into fixed space:

  --labels  → config.MOVING_INSTANCES, nearestNeighbor interp (preserves IDs)
  --raw     → config.MOVING_RAW,       linear interp       (preserves intensity)

Both flags default to True; pass --no-labels or --no-raw to skip one.

Outputs (config.WARPED):
  moving_labels_warped.zarr   (with --labels)
  moving_raw_warped.zarr      (with --raw and MOVING_RAW configured)
"""

import json
import logging

import ants
import click
import numpy as np

import config
from utils import load_volume, load_zarr, save_volume, timed, to_ants

log = logging.getLogger(__name__)


def _load_fwd_chain():
    manifest_path = config.TRANSFORMS / 'transform_manifest.json'
    assert manifest_path.exists(), (
        'Missing transform_manifest.json — run stage5_affine (or stage5b_syn) first'
    )
    fwd = json.loads(manifest_path.read_text())['fwdtransforms']
    log.info('Forward chain: %s', fwd)
    return fwd


def _warp_source(src_spec, src_spacing_nm, fixed_ants, fwd,
                 interpolator, name):
    """
    Load a moving source, apply the forward chain, return (warped, src).
    Both arrays are numpy; warped is cast back to src.dtype.
    """
    log.info('[%s] loading source …', name)
    src = load_zarr(src_spec)
    src_dtype = src.dtype
    if np.issubdtype(src_dtype, np.integer) and src.max() > 2 ** 24:
        log.warning('[%s] integer IDs > 2^24 — the float32 round-trip through '
                    'ANTs will lose precision (max=%d)', name, int(src.max()))

    src_ants = to_ants(src.astype(np.float32), src_spacing_nm)
    with timed(f'[{name}] apply_transforms ({interpolator})'):
        warped = ants.apply_transforms(
            fixed=fixed_ants, moving=src_ants,
            transformlist=fwd, interpolator=interpolator,
        )
    arr = warped.numpy()
    if np.issubdtype(src_dtype, np.integer):
        arr = np.round(arr).astype(src_dtype)
    else:
        arr = arr.astype(src_dtype)
    log.info('[%s] warped: shape=%s dtype=%s', name, arr.shape, arr.dtype)
    return arr, src


def run(do_labels: bool = True, do_raw: bool = True):
    config.WARPED.mkdir(parents=True, exist_ok=True)
    fwd = _load_fwd_chain()

    # The fixed reference defines the output grid (shape, spacing, origin).
    with timed('load fixed_binary (reference grid)'):
        fixed_arr = load_volume(config.INTERMEDIATE / 'fixed_binary.zarr')
    fixed_ants = to_ants(fixed_arr.astype(np.float32), config.FIXED_SPACING_NM)
    del fixed_arr   # only its geometry is needed downstream

    if do_labels:
        # nearestNeighbor is ~10× faster than genericLabel on volumes this big
        # (~3.7B output voxels). genericLabel preserves label topology better
        # but the compute is often impractical; swap back here if you need it.
        warped, src = _warp_source(
            config.MOVING_INSTANCES, config.MOVING_SPACING_NM,
            fixed_ants, fwd, 'nearestNeighbor', 'labels',
        )
        save_volume(config.WARPED / 'moving_labels_warped.zarr', warped)

        in_labels  = set(np.unique(src[src > 0]).tolist())
        out_labels = set(np.unique(warped[warped > 0]).tolist())
        lost = len(in_labels - out_labels)
        frac = lost / max(len(in_labels), 1)
        log.info('[labels] in=%d  out=%d  lost=%d  (%.2f%%)',
                 len(in_labels), len(out_labels), lost, 100 * frac)
        if frac > config.LABEL_LOSS_FRACTION_WARN:
            log.warning('[labels] label loss %.2f%% exceeds threshold %.2f%%',
                        100 * frac, 100 * config.LABEL_LOSS_FRACTION_WARN)

    if do_raw:
        if not getattr(config, 'MOVING_RAW', None):
            log.warning('config.MOVING_RAW is not configured — skipping raw '
                        'warp. Set MOVING_RAW = {"path": ..., "component": ...} '
                        'in config.py to enable.')
        else:
            warped, _ = _warp_source(
                config.MOVING_RAW, config.MOVING_SPACING_NM,
                fixed_ants, fwd, 'linear', 'raw',
            )
            save_volume(config.WARPED / 'moving_raw_warped.zarr', warped)

    log.info('Stage 7 complete.')


@click.command()
@click.option('--labels/--no-labels', default=True,
              help='Warp moving instance segmentation (nearestNeighbor interp).')
@click.option('--raw/--no-raw', default=True,
              help='Warp raw moving intensity volume (linear interp).')
def main(labels, raw):
    run(do_labels=labels, do_raw=raw)


if __name__ == '__main__':
    main()

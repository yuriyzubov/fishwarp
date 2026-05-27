"""
Stage 0: Stream-binarize instance segmentations from zarr and save uint8 binary
masks. Processes chunks in parallel via dask's threaded scheduler — no
full-resolution arrays held in RAM at any point.

Outputs:
  outputs/intermediate/fixed_binary.zarr
  outputs/intermediate/moving_binary.zarr
  outputs/intermediate/manifest.json   (zarr specs for Stage 7)
"""

import json
import logging

import config
from utils import binarize_to_zarr, timed

log = logging.getLogger(__name__)


def run():
    config.INTERMEDIATE.mkdir(parents=True, exist_ok=True)

    n_workers = config.STAGE0_N_WORKERS

    with timed('binarize fixed'):
        binarize_to_zarr(
            config.FIXED_INSTANCES,
            config.INTERMEDIATE / 'fixed_binary.zarr',
            name='fixed',
            pad=config.FIXED_PAD_VOXELS,
            n_workers=n_workers,
        )

    with timed('binarize moving'):
        binarize_to_zarr(
            config.MOVING_INSTANCES,
            config.INTERMEDIATE / 'moving_binary.zarr',
            name='moving',
            pad=0,
            n_workers=n_workers,
        )

    manifest = {
        'fixed_instances':   config.FIXED_INSTANCES,
        'moving_instances':  config.MOVING_INSTANCES,
        'fixed_spacing_nm':  list(config.FIXED_SPACING_NM),
        'moving_spacing_nm': list(config.MOVING_SPACING_NM),
    }
    manifest_path = config.INTERMEDIATE / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info('Manifest saved to %s', manifest_path)

    log.info('Stage 0 complete.')


if __name__ == '__main__':
    run()

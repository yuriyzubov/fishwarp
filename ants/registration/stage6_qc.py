"""
Stage 6: Quality control — metrics, slice PNGs, and neuroglancer viewer.

Loads fixed binary and applies the full affine transform chain to the moving
binary mask, then computes Dice and SDF-based metrics and launches a
neuroglancer viewer with three layers.

Zarr intermediates are served to neuroglancer directly (no full RAM load).
Only the warped moving binary requires materialisation in memory (ANTs output).

Outputs:
  outputs/qc/metrics.json
  outputs/qc/slice_z.png
  outputs/qc/slice_y.png
  outputs/qc/slice_x.png
"""

import logging

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    intersection = (a & b).sum()
    return 2 * intersection / (a.sum() + b.sum() + 1e-9)

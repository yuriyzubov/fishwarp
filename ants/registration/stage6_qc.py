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


def mean_abs_sdf_diff(fixed_sdf: np.ndarray, warped_sdf: np.ndarray,
                      overlap_mask: np.ndarray) -> float:
    diff = np.abs(fixed_sdf[overlap_mask] - warped_sdf[overlap_mask])
    return float(diff.mean())


def bbox_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of bounding boxes of foreground regions."""
    def bbox(arr):
        idx = np.argwhere(arr)
        return idx.min(axis=0), idx.max(axis=0)

    lo_a, hi_a = bbox(a)
    lo_b, hi_b = bbox(b)
    lo_i = np.maximum(lo_a, lo_b)
    hi_i = np.minimum(hi_a, hi_b)
    if np.any(hi_i < lo_i):
        return 0.0
    def vol(lo, hi): return np.prod(hi - lo + 1)
    inter = vol(lo_i, hi_i)
    union = vol(lo_a, hi_a) + vol(lo_b, hi_b) - inter
    return float(inter / union)

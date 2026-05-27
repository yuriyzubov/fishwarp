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

import json
import logging

import matplotlib.pyplot as plt
import neuroglancer
import numpy as np
import zarr

import config
from utils import (
    load_volume, make_neuroglancer_dims,
    add_segmentation_layer, serve_viewer, timed,
)

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


# ---------------------------------------------------------------------------
# Slice PNGs
# ---------------------------------------------------------------------------

def save_slice_png(fixed: np.ndarray, warped: np.ndarray,
                   axis: int, out_path) -> None:
    mid = fixed.shape[axis] // 2
    sl = [slice(None)] * 3
    sl[axis] = mid
    sl = tuple(sl)
    f = fixed[sl].astype(float)
    w = warped[sl].astype(float)

    rgb = np.zeros((*f.shape, 3))
    rgb[..., 0] = np.clip(f, 0, 1)          # fixed → red
    rgb[..., 1] = np.clip(w, 0, 1)          # warped → green
    # overlap → yellow (both channels lit)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb, origin='lower', interpolation='nearest')
    ax.axis('off')
    axis_name = ['z', 'y', 'x'][axis]
    ax.set_title(f'mid-{axis_name} slice  (red=fixed, green=warped, yellow=overlap)')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info('Saved slice PNG: %s', out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    config.QC.mkdir(parents=True, exist_ok=True)

    # -- Load transform manifest -----------------------------------------
    manifest_path = config.TRANSFORMS / 'transform_manifest.json'
    assert manifest_path.exists(), f'Missing transform manifest: {manifest_path}'
    manifest = json.loads(manifest_path.read_text())
    fwd_transforms = manifest['fwdtransforms']
    log.info('Transform chain: %s', fwd_transforms)

    # -- Load fixed binary (numpy — needed for metrics) ------------------
    with timed('load fixed binary'):
        fixed_binary = load_volume(config.INTERMEDIATE / 'fixed_binary.zarr')
    assert fixed_binary.ndim == 3

    # -- Load fixed + warped SDFs if available (SDF metric is optional) ---
    fixed_sdf_path = (
        config.INTERMEDIATE / 'fixed_sdf_blurred.zarr'
        if config.ENABLE_STAGE2_BLUR
        else config.INTERMEDIATE / 'fixed_sdf.zarr'
    )
    warped_sdf_path = config.INTERMEDIATE / 'moving_sdf_warped_affine.zarr'
    sdf_available = fixed_sdf_path.exists() and warped_sdf_path.exists()
    if sdf_available:
        with timed('load fixed SDF'):
            fixed_sdf = load_volume(fixed_sdf_path)
        with timed('load warped SDF'):
            warped_sdf = load_volume(warped_sdf_path)
    else:
        log.info('SDFs not on disk — skipping SDF-based metric (REG_IMAGE_TYPE=%s).',
                 config.REG_IMAGE_TYPE)
        fixed_sdf = warped_sdf = None

    # -- Load warped binary (saved by whichever stage ran last) ----------
    # Priority: SyN refine > SyN > affine > rigid > COM — avoids recomputing
    # apply_transforms.
    for candidate in ('moving_binary_warped_syn_refine.zarr',
                      'moving_binary_warped_syn.zarr',
                      'moving_binary_warped_affine.zarr',
                      'moving_binary_warped_rigid.zarr',
                      'moving_binary_warped_com.zarr'):
        warped_path = config.INTERMEDIATE / candidate
        if warped_path.exists():
            log.info('Loading warped binary from %s', candidate)
            warped_binary = load_volume(warped_path)
            break
    else:
        raise FileNotFoundError(
            'No warped binary found. Run at least stage3_com first.'
        )

    # -- Metrics ---------------------------------------------------------
    with timed('compute metrics'):
        d = dice(fixed_binary, warped_binary)
        bb_iou = bbox_overlap(fixed_binary, warped_binary)
        sdf_diff = None
        if sdf_available:
            overlap_mask = (fixed_binary > 0) & (warped_binary > 0)
            sdf_diff = mean_abs_sdf_diff(fixed_sdf, warped_sdf, overlap_mask)

    metrics = {
        'dice':     round(d, 4),
        'bbox_iou': round(bb_iou, 4),
    }
    if sdf_diff is not None:
        metrics['mean_abs_sdf_diff'] = round(sdf_diff, 6)
    (config.QC / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    log.info('Metrics: %s', metrics)

    if d >= config.DICE_GOOD:
        verdict = 'GOOD'
    elif d >= config.DICE_MARGINAL:
        verdict = 'MARGINAL — consider deformable (stage 5)'
    else:
        verdict = 'POOR — run deformable (stage 5)'
    log.info('Verdict: Dice=%.4f → %s', d, verdict)

    # -- Slice PNGs ------------------------------------------------------
    for axis in range(3):
        save_slice_png(
            fixed_binary, warped_binary, axis,
            config.QC / f'slice_{"zyx"[axis]}.png',
        )

    # -- Neuroglancer viewer ---------------------------------------------
    neuroglancer.set_server_bind_address(config.NEUROGLANCER_BIND_ADDRESS)
    viewer = neuroglancer.Viewer()
    dims = make_neuroglancer_dims(config.FIXED_SPACING_NM)

    # Open zarr arrays lazily — served chunk-by-demand, no full RAM load
    fixed_zarr        = zarr.open_array(str(config.INTERMEDIATE / 'fixed_binary.zarr'),  mode='r')
    moving_unw_zarr   = zarr.open_array(str(config.INTERMEDIATE / 'moving_binary.zarr'), mode='r')

    with viewer.txn() as s:
        s.dimensions = dims
        add_segmentation_layer(s, 'fixed',           fixed_zarr,    dims, visible=True)
        add_segmentation_layer(s, 'moving_warped',   warped_binary, dims, visible=True)
        add_segmentation_layer(s, 'moving_unwarped', moving_unw_zarr, dims, visible=False)

    serve_viewer(viewer)


if __name__ == '__main__':
    run()

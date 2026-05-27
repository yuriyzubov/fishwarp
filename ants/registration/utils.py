"""
Shared helpers: I/O, ANTs conversion, neuroglancer setup, logging, timing.
"""

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging — must be configured before any third-party imports, which may
# configure the root logger themselves and make basicConfig() a no-op.
# ---------------------------------------------------------------------------

_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter(
    '%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
))
logging.root.setLevel(logging.INFO)
logging.root.addHandler(_handler)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ANTs / ITK threading — set before importing ants so ITK picks it up.
# Respects any value already set in the environment.
# ---------------------------------------------------------------------------

_n_threads = os.environ.setdefault(
    'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', str(os.cpu_count())
)
log.info('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS = %s', _n_threads)

import ants
import numcodecs
import numpy as np
import zarr

_COMPRESSOR = numcodecs.Zstd(level=3)
_CHUNKS     = (128, 128, 128)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

@contextmanager
def timed(label: str):
    """Context manager that logs elapsed time for a block."""
    log.info('START  %s', label)
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    log.info('DONE   %s  (%.1f s)', label, elapsed)


# ---------------------------------------------------------------------------
# Zarr I/O
# ---------------------------------------------------------------------------

def load_zarr(spec: dict) -> np.ndarray:
    """
    Open a zarr store, navigate to the component if specified, and return
    the array as a numpy ndarray.  Prints attrs, shape, dtype, and chunks
    for sanity-checking.  Fails loudly if the array has more than 3 dimensions.
    """
    path = spec['path']
    component = spec.get('component')

    log.info('Opening zarr store: %s', path)
    store = zarr.open(path, mode='r')

    if component:
        log.info('Navigating to component: %s', component)
        arr = store[component]
    else:
        arr = store

    # Print metadata before loading
    attrs = dict(arr.attrs) if hasattr(arr, 'attrs') else {}
    log.info('  .attrs   : %s', json.dumps(attrs, indent=2, default=str))
    log.info('  .shape   : %s', arr.shape)
    log.info('  .dtype   : %s', arr.dtype)
    log.info('  .chunks  : %s', getattr(arr, 'chunks', 'N/A'))

    if arr.ndim != 3:
        raise ValueError(
            f"Array at '{path}'/'{component}' has {arr.ndim} dimensions (shape={arr.shape}). "
            "Expected exactly 3 (z, y, x). Please specify which axes to use."
        )

    log.info('Loading into RAM …')
    with timed('zarr → numpy'):
        data = arr[:]

    log.info('Loaded array: shape=%s  dtype=%s', data.shape, data.dtype)
    return data


# ---------------------------------------------------------------------------
# ANTs conversion
# ---------------------------------------------------------------------------
#
# Axis ordering note:
#   numpy arrays in this pipeline are always (z, y, x).
#   ants.from_numpy() applies spacing[k] to numpy axis k — it does NOT reverse
#   axes. So spacing must be passed in the SAME (z, y, x) order as the array.
#   (An earlier version reversed the tuple; that silently swapped the z and x
#   spacings, badly distorting anisotropic volumes like the confocal moving
#   image — 1000/259/259 nm became a 3.86x-stretched pancake.)

def to_ants(array: np.ndarray, spacing_nm: tuple) -> ants.ANTsImage:
    """
    Wrap a (z, y, x) numpy array as an ANTs image.
    spacing_nm must be a (z, y, x) tuple in nanometers; ANTs receives it in
    the same axis order, converted to mm.
    """
    spacing_mm = tuple(s / 1e6 for s in spacing_nm)  # (z, y, x) mm
    return ants.from_numpy(array.astype(np.float32), spacing=spacing_mm)


def from_ants(ants_image: ants.ANTsImage) -> np.ndarray:
    """Return the numpy array backing an ANTs image (z, y, x ordering preserved)."""
    return ants_image.numpy()


# ---------------------------------------------------------------------------
# Intermediate volume I/O (zarr, zstd-compressed, chunk-aligned)
# ---------------------------------------------------------------------------

def save_volume(path: Path, array: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    z = zarr.open_array(
        str(path),
        mode='w',
        shape=array.shape,
        chunks=_CHUNKS,
        dtype=array.dtype,
        compressor=_COMPRESSOR,
        dimension_separator='/',
    )
    z[:] = array
    log.info('Saved %s  shape=%s  dtype=%s', path, array.shape, array.dtype)


def load_volume(path: Path) -> np.ndarray:
    arr = zarr.open_array(str(path), mode='r')[:]
    log.info('Loaded %s  shape=%s  dtype=%s', path, arr.shape, arr.dtype)
    return arr


# ---------------------------------------------------------------------------
# Chunked streaming binarize: read source zarr block-by-block, threshold to
# uint8, write to destination zarr. Never holds the full array in RAM.
# Parallelized with dask.delayed + the threaded scheduler — zarr decompression
# and numpy ufuncs release the GIL, so threads scale across cores here.
# ---------------------------------------------------------------------------

def binarize_to_zarr(
    src_spec: dict,
    dst_path: Path,
    name: str = '',
    pad: int = 0,
    n_workers: int = 16,
    block_shape: tuple = _CHUNKS,
) -> dict:
    """
    Stream-binarize a 3D instance segmentation zarr into a uint8 binary zarr.

    Iterates over `block_shape`-aligned slabs of the source, dispatches each
    block (read → `>0` → write) as a dask.delayed task, then runs them all
    via dask.compute on the threaded scheduler.

    pad > 0 grows the destination shape by 2*pad on each axis and offsets
    writes by `pad`. Border voxels keep the zarr fill value (0).
    """
    import dask
    from dask import delayed

    src_path = src_spec['path']
    component = src_spec.get('component')

    src_root = zarr.open(src_path, mode='r')
    src = src_root[component] if component else src_root

    if src.ndim != 3:
        raise ValueError(
            f"Source at {src_path}/{component} is not 3D (shape={src.shape})"
        )

    log.info('[%s] source: shape=%s dtype=%s chunks=%s',
             name, src.shape, src.dtype, getattr(src, 'chunks', None))

    out_shape = tuple(s + 2 * pad for s in src.shape)
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    dst = zarr.open_array(
        str(dst_path),
        mode='w',
        shape=out_shape,
        chunks=block_shape,
        dtype=np.uint8,
        compressor=_COMPRESSOR,
        dimension_separator='/',
        fill_value=0,
    )
    log.info('[%s] dest:   shape=%s chunks=%s pad=%d',
             name, out_shape, block_shape, pad)

    bz, by, bx = block_shape
    sz, sy, sx = src.shape
    slabs = [
        (
            slice(z0, min(z0 + bz, sz)),
            slice(y0, min(y0 + by, sy)),
            slice(x0, min(x0 + bx, sx)),
        )
        for z0 in range(0, sz, bz)
        for y0 in range(0, sy, by)
        for x0 in range(0, sx, bx)
    ]
    log.info('[%s] dispatching %d blocks via dask (threads=%d) …',
             name, len(slabs), n_workers)

    @delayed
    def _process(slab):
        block = src[slab]
        binary = (block > 0).astype(np.uint8)
        if pad:
            out_slab = tuple(slice(s.start + pad, s.stop + pad) for s in slab)
        else:
            out_slab = slab
        dst[out_slab] = binary
        return int(binary.sum()), binary.size

    tasks = [_process(slab) for slab in slabs]

    t0 = time.perf_counter()
    results = dask.compute(*tasks, scheduler='threads', num_workers=n_workers)
    elapsed = time.perf_counter() - t0

    total_fg = sum(r[0] for r in results)
    total_n  = sum(r[1] for r in results)
    occ = total_fg / total_n if total_n else 0.0
    log.info('[%s] done in %.1fs — foreground=%d / %d  occupancy=%.3f',
             name, elapsed, total_fg, total_n, occ)

    return {
        'src_shape': tuple(src.shape),
        'out_shape': out_shape,
        'foreground': total_fg,
        'n_voxels': total_n,
        'occupancy': occ,
        'elapsed_s': elapsed,
    }


# ---------------------------------------------------------------------------
# Chunked resample onto a common grid: tile the OUTPUT grid, and for each tile
# read only the small source slab it needs (output bbox mapped back to source
# index space + halo), resample that slab with ANTs, write into the output
# array. Never materializes the full source in RAM — same memory win as
# binarize_to_zarr, but with a halo because resampling is a gather, not an
# elementwise map. Parallelized with dask.delayed + the threaded scheduler
# (zarr decompression and ITK resampling release the GIL).
#
# Bit-identical to ants.resample_image_to_target(src, ref, 'linear') where ref
# has origin (0,0,0) and isotropic spacing: each tile builds genuine ANTs
# sub-images with the correct physical origin, so ANTs maps coordinates the
# same way it would for the whole volume.
# ---------------------------------------------------------------------------

def resample_to_grid_chunked(
    src_zarr,
    src_spacing_mm: tuple,
    target_shape: tuple,
    target_spacing_mm: float,
    name: str = '',
    n_workers: int = 16,
    block_shape: tuple = _CHUNKS,
    halo: int = 2,
) -> np.ndarray:
    """
    Resample a 3D source zarr onto a common grid (origin (0,0,0), isotropic
    `target_spacing_mm`, shape `target_shape`) with linear interpolation.

    `src_spacing_mm` is the per-numpy-axis spacing of the source in mm — i.e.
    the same tuple passed to ants.from_numpy(..., spacing=...) for the source.
    Source axis a corresponds to output axis a (matching ANTs dimension).

    Returns a float32 numpy array of shape `target_shape`.
    """
    import dask
    from dask import delayed

    src_spacing_mm = tuple(float(s) for s in src_spacing_mm)
    # input voxels per output voxel, per axis
    ratio = tuple(target_spacing_mm / s for s in src_spacing_mm)
    out = np.zeros(target_shape, dtype=np.float32)

    tz, ty, tx = target_shape
    bz, by, bx = block_shape
    tiles = [
        (
            slice(z0, min(z0 + bz, tz)),
            slice(y0, min(y0 + by, ty)),
            slice(x0, min(x0 + bx, tx)),
        )
        for z0 in range(0, tz, bz)
        for y0 in range(0, ty, by)
        for x0 in range(0, tx, bx)
    ]
    log.info('[%s] resample: source=%s → target=%s  ratio=%s  '
             'dispatching %d tiles via dask (threads=%d) …',
             name, tuple(src_zarr.shape), tuple(target_shape),
             tuple(round(r, 3) for r in ratio), len(tiles), n_workers)

    @delayed
    def _process(tile):
        # Map the output tile's index range back to source index space.
        in_sl = []
        for a, o in enumerate(tile):
            lo = int(np.floor(o.start * ratio[a])) - halo
            hi = int(np.ceil((o.stop - 1) * ratio[a])) + 1 + halo
            lo = max(lo, 0)
            hi = min(hi, src_zarr.shape[a])
            if hi <= lo:
                return 0.0          # tile lies entirely outside the source → 0
            in_sl.append(slice(lo, hi))
        in_sl = tuple(in_sl)

        block = np.asarray(src_zarr[in_sl], dtype=np.float32)
        src_origin = tuple(in_sl[a].start * src_spacing_mm[a] for a in range(3))
        src_ants = ants.from_numpy(block, origin=src_origin,
                                   spacing=src_spacing_mm)

        tgt_shape  = tuple(o.stop - o.start for o in tile)
        tgt_origin = tuple(tile[a].start * target_spacing_mm for a in range(3))
        tgt_ants = ants.from_numpy(
            np.zeros(tgt_shape, dtype=np.float32),
            origin=tgt_origin,
            spacing=(target_spacing_mm,) * 3,
        )

        res = ants.resample_image_to_target(
            src_ants, tgt_ants, interp_type='linear',
        ).numpy()
        out[tile] = res
        return float(res.sum())

    tasks = [_process(tile) for tile in tiles]

    t0 = time.perf_counter()
    results = dask.compute(*tasks, scheduler='threads', num_workers=n_workers)
    elapsed = time.perf_counter() - t0

    log.info('[%s] resample done in %.1fs — output shape=%s mean=%.4f',
             name, elapsed, out.shape, float(out.mean()))
    return out


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------

def quick_dice(a: np.ndarray, b: np.ndarray) -> float:
    """Fast numpy Dice coefficient between two binary arrays."""
    a, b = a.astype(bool), b.astype(bool)
    return float(2 * (a & b).sum() / (a.sum() + b.sum() + 1e-9))


def warp_binary(moving_binary: np.ndarray, moving_spacing_nm: tuple,
                fixed_binary: np.ndarray, fixed_spacing_nm: tuple,
                transformlist: list) -> np.ndarray:
    """
    Apply a transform chain to a binary mask using nearestNeighbor interpolation.
    Returns uint8 array in fixed space.
    """
    fixed_ants  = to_ants(fixed_binary.astype(np.float32),  fixed_spacing_nm)
    moving_ants = to_ants(moving_binary.astype(np.float32), moving_spacing_nm)
    warped = ants.apply_transforms(
        fixed=fixed_ants,
        moving=moving_ants,
        transformlist=transformlist,
        interpolator='nearestNeighbor',
    )
    return (from_ants(warped) > 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def downsample(ants_image: ants.ANTsImage, factor: int) -> ants.ANTsImage:
    """
    Downsample an ANTs image by an integer factor using block averaging.
    Each output voxel is the mean of a factor³ input block (anti-aliased, fast).
    Input is cropped to the nearest multiple of factor before averaging.
    """
    arr = ants_image.numpy()
    z, y, x = arr.shape
    # Crop to nearest multiple of factor so reshape is exact
    arr = arr[:z - z % factor, :y - y % factor, :x - x % factor]
    z, y, x = arr.shape
    arr_ds = (arr.reshape(z // factor, factor,
                          y // factor, factor,
                          x // factor, factor)
                 .mean(axis=(1, 3, 5))
                 .astype(np.float32))
    new_spacing = tuple(s * factor for s in ants_image.spacing)
    return ants.from_numpy(arr_ds, spacing=new_spacing)

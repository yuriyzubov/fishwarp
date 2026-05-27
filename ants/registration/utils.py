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

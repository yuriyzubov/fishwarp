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

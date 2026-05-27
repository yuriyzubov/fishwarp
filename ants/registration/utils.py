"""
Shared helpers: I/O, ANTs conversion, neuroglancer setup, logging, timing.
"""

import logging
import os
import sys

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

import numcodecs

_COMPRESSOR = numcodecs.Zstd(level=3)
_CHUNKS     = (128, 128, 128)

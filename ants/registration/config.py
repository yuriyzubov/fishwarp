"""
All paths, voxel spacings, and tunable parameters for the registration pipeline.
Edit this file before running any stage.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Input zarr specs — fill in before running
# ---------------------------------------------------------------------------

FIXED_INSTANCES = {
    'path': '/path/to/fixed_instances.zarr',       # EDIT: path to fixed (e.g. EM) instance-segmentation zarr
    'component': 's0',                             # EDIT: internal path (e.g. 'volumes/labels/nuclei') or None for root
}

MOVING_INSTANCES = {
    'path': '/path/to/moving_instances.zarr',      # EDIT: path to moving (e.g. confocal) instance-segmentation zarr
    'component': 's0',                             # EDIT: internal path or None for root
}

# Optional: raw moving intensity volume — used by stage 7 --raw. Leave as
# None to skip the raw warp; otherwise fill in {path, component}.
MOVING_RAW = {
    'path': '/path/to/moving_raw.zarr',            # EDIT
    'component': 's0',                             # EDIT: internal path or None
}

# ---------------------------------------------------------------------------
# Voxel spacings in nanometers, as (z, y, x) tuples
# ---------------------------------------------------------------------------

FIXED_SPACING_NM  = (480, 512, 512)    # EDIT: fixed-volume voxel spacing
MOVING_SPACING_NM = (1000, 259, 259)   # EDIT: moving-volume voxel spacing

# ---------------------------------------------------------------------------
# Output directories — fill in before running
# ---------------------------------------------------------------------------

OUTPUT_ROOT   = Path('/path/to/output_root')   # EDIT
INTERMEDIATE  = OUTPUT_ROOT / 'intermediate'
TRANSFORMS    = OUTPUT_ROOT / 'transforms'
WARPED        = OUTPUT_ROOT / 'warped'
QC            = OUTPUT_ROOT / 'qc'

# ---------------------------------------------------------------------------
# Stage 0 / 1b (chunked dask scheduler)
# ---------------------------------------------------------------------------

STAGE0_N_WORKERS  = 16    # chunked binarize — workers for chunked read/write
STAGE1B_N_WORKERS = 16    # chunked resample — workers for tiled resample-to-grid

# ---------------------------------------------------------------------------
# Stage flags
# ---------------------------------------------------------------------------

ENABLE_STAGE2_BLUR = False  # PSF blur: disabled — sigma is negligible for these spacings

# ---------------------------------------------------------------------------
# Distance transform
# ---------------------------------------------------------------------------

SDF_CLIP_VOXELS   = 200     # clip SDF to ±N voxels (applied after EDT, before saving)
EDT_PARALLEL_JOBS = 8       # threads for edt.edt()

# ---------------------------------------------------------------------------
# Registration image type (controls what stages 4/5 register)
# ---------------------------------------------------------------------------
#   'sdf'  — clipped signed distance fields at native spacings (stage1_distance.py)
#   'blur' — Gaussian-blurred binary masks resampled to a shared isotropic grid
#            with origin (0,0,0) (stage1b_blur.py). Closer to the working
#            register_and_warp.py density-image setup; usually more stable for
#            ANTs rigid + affine when fixed/moving differ in shape and spacing.
REG_IMAGE_TYPE         = 'blur'  # 'sdf' | 'blur'
COMMON_GRID_SPACING_UM = 2.0     # used only when REG_IMAGE_TYPE == 'blur'
BLUR_SIGMA_UM          = 2.0     # used only when REG_IMAGE_TYPE == 'blur'

# ---------------------------------------------------------------------------
# Registration — affine (stage 3/4)
# ---------------------------------------------------------------------------

AFFINE_DOWNSAMPLE_FACTOR = 2   # integer factor applied to each axis before affine

# Symmetric zero-padding (in voxels) added to the fixed binary in stage 0 so
# that warped moving masks aren't clipped when their extent exceeds the fixed
# bounding box. Applied per-axis to both sides; affects all downstream stages.
FIXED_PAD_VOXELS = 0

# ---------------------------------------------------------------------------
# Registration — SyN deformable (stage 5)
# ---------------------------------------------------------------------------

SYN_DOWNSAMPLE_FACTOR = 1      # no downsampling for deformable
SYN_FLOW_SIGMA        = 5.0    # stiff regularization — do not relax
SYN_TOTAL_SIGMA       = 1.0
SYN_ITERATIONS        = (40, 30, 30)

# ---------------------------------------------------------------------------
# Neuroglancer
# ---------------------------------------------------------------------------

NEUROGLANCER_BIND_ADDRESS = '127.0.0.1'
NEUROGLANCER_PORT         = 0  # 0 = let neuroglancer pick a free port

# ---------------------------------------------------------------------------
# QC thresholds
# ---------------------------------------------------------------------------

DICE_GOOD     = 0.85
DICE_MARGINAL = 0.70

# ---------------------------------------------------------------------------
# Stage 7 — label preservation tolerance
# ---------------------------------------------------------------------------

LABEL_LOSS_FRACTION_WARN = 0.05  # warn if > 5% of moving labels disappear after warping

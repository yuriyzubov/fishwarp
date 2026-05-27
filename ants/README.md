# ants/registration — pairwise volume registration pipeline

Registers a 3D moving instance segmentation onto a 3D fixed instance
segmentation via COM → rigid → affine → SyN (deformable), then warps the
moving labels and raw intensity volumes into fixed space.

## Install

```bash
pip install ants/                  # from the repo root
# or: cd ants && pip install .
```

Requires Python ≥ 3.11. Pulls in antspyx, dask, edt, matplotlib,
neuroglancer, numcodecs, numpy, zarr.

## Configure

Edit `ants/registration/config.py`:

- `FIXED_INSTANCES`, `MOVING_INSTANCES` — zarr `{path, component}` for the
  two instance segmentations. `component` is the internal array path
  (e.g. `'s0'`, `'volumes/labels/nuclei'`) or `None` for the root.
- `MOVING_RAW` — optional; raw intensity zarr to warp at stage 7.
- `FIXED_SPACING_NM`, `MOVING_SPACING_NM` — voxel spacing as `(z, y, x)`
  nanometer tuples.
- `OUTPUT_ROOT` — everything (intermediates, transforms, warped, QC)
  lands here.
- `REG_IMAGE_TYPE` — `'blur'` (default, recommended) or `'sdf'`. Blur
  resamples both masks to a common isotropic grid and Gaussian-blurs
  them; SDF builds anisotropic signed distance fields at native spacing.
  Blur is more stable for ANTs rigid + affine when fixed and moving have
  different shapes and spacings.

Everything else in `config.py` has working defaults — only retune if you
have a reason.

## Run

All stages assume the working directory is `ants/registration/` (they do
`import config` / `from utils import …`).

```bash
cd ants/registration

python stage0_load.py                 # binarize fixed + moving → fixed_binary.zarr, moving_binary.zarr
python stage1b_blur.py                # resample + blur on common grid (REG_IMAGE_TYPE='blur')
# or, for REG_IMAGE_TYPE='sdf':
# python stage1_distance.py

python stage3_com.py --warp           # COM translation + warped binary QC
python stage4_rigid.py  --warp        # rigid, initialized from COM
python stage5_affine.py --warp        # affine, initialized from rigid

python stage5b_syn.py     --warp      # SyN deformable on top of affine
python stage5c_syn_refine.py --warp   # optional: stack additional SyN passes
                                      # (re-run as many times as needed; each
                                      #  refinement composes onto the chain)

python stage6_qc.py                   # Dice + bbox IoU + slice PNGs + neuroglancer
python stage7_warp.py                 # apply chain to labels and raw
```

`--warp` on stages 3/4/5/5b/5c is optional — it applies the chain to the
moving binary and saves the warped zarr so you can inspect intermediate
alignment. Skip it on long runs; stage 6 will pick up the latest
available warped binary regardless.

## Outputs

Under `OUTPUT_ROOT`:

- `intermediate/` — binaries, blur/SDF images, warped binaries from each
  stage.
- `transforms/` — `com_translation.mat`, `rigid.mat`, `affine_0GenericAffine.mat`,
  `syn_*Warp.nii.gz`, and `transform_manifest.json` (the live forward/
  inverse chain that stages 5b/5c/6/7 read).
- `qc/` — `metrics.json` (Dice, bbox IoU), `slice_{z,y,x}.png` (red =
  fixed, green = warped, yellow = overlap), `stage3_extents.log`.
- `warped/` — `moving_labels_warped.zarr`, `moving_raw_warped.zarr`.

## Reading the QC verdict

Stage 6 prints a Dice-based verdict using the thresholds in config:
- Dice ≥ `DICE_GOOD` (0.85) — GOOD.
- Dice ≥ `DICE_MARGINAL` (0.70) — MARGINAL; consider deformable.
- Dice < 0.70 — POOR; run deformable.

If the affine is already MARGINAL, run `stage5b_syn.py` and then
`stage6_qc.py` again. If post-SyN Dice is still off in a specific
region, run `stage5c_syn_refine.py` to add another deformable pass on
top — the chain is composed explicitly so multiple refinements stack
cleanly.

## Troubleshooting

- **Stage 4 fails with "images do not sufficiently overlap"** — the COM
  init didn't bring the foregrounds together. Run `python diag_overlap.py`
  to inspect blur image stats, COM coordinates, and per-axis overlap.
  Common causes: wrong voxel spacing tuple, axis order mismatch between
  fixed and moving, or one volume being mostly background.
- **Label loss warning in stage 7** — `LABEL_LOSS_FRACTION_WARN` (5%) is
  the threshold for the count of distinct label IDs lost during NN-interp
  resampling. If your labels are larger than the moving voxel, this is
  normally well below threshold.
- **Integer labels > 2^24** — stage 7 logs a warning. The float32
  round-trip through ANTs loses precision for IDs above 16,777,216. If
  this matters, split the label volume or remap IDs before warping.

## File map

```
ants/
├── pyproject.toml              # dependencies for the pipeline
├── README.md                   # this file
└── registration/
    ├── config.py               # paths, spacings, tunables
    ├── utils.py                # shared I/O, ANTs wrap/unwrap, neuroglancer helpers
    ├── stage0_load.py          # chunked binarize → uint8 zarr
    ├── stage1_distance.py      # signed distance transform (sdf mode)
    ├── stage1b_blur.py         # common-grid resample + Gaussian blur (blur mode)
    ├── stage3_com.py           # center-of-mass translation init
    ├── stage4_rigid.py         # rigid registration
    ├── stage5_affine.py        # affine registration
    ├── stage5b_syn.py          # SyN deformable
    ├── stage5c_syn_refine.py   # iterative SyN refinement
    ├── stage6_qc.py            # metrics, slice PNGs, neuroglancer viewer
    ├── stage7_warp.py          # apply chain to labels and raw
    └── diag_overlap.py         # diagnose stage 4 overlap failures
```

# FishWarp: Zebrafish Image Analysis Pipeline

A comprehensive pipeline for zebrafish image analysis including cell segmentation using Cellpose and image registration using BigStream.

## Overview

FishWarp provides tools for:
- **Cell Segmentation**: Distributed Cellpose segmentation with foreground masking
- **Image Registration**: BigStream-based registration with multiple similarity metrics
- **Quality Assessment**: Registration quality metrics including NCC, mutual information, and MSE

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, for Cellpose acceleration)

### Dependencies
Create a conda/mamba environment:
```bash
mamba env create -f environment.yml
# or
conda env create -f environment.yml
```

Key dependencies include:
- `cellpose`
- `zarr`
- `bigstream`
- `SimpleITK`
- `scikit-image`
- `scipy`
- `numpy`

## Directory Structure

```
fishwarp/
├── cellpose/          # Cellpose segmentation modules
│   ├── utils.py       # Core segmentation functions
│   ├── get_segmentation.ipynb  # Interactive segmentation notebook
│   └── config_example.json     # Configuration template
└── bigstream/         # Image registration modules
    └── metrics.py     # Registration quality metrics
```

## Usage

### 1. Cell Segmentation

#### Configuration
Create a JSON configuration file based on `config_example.json`:

```json
{
  "src_path": "/path/to/input.zarr/s0",
  "dest_path": "/path/to/output/directory",
  "foreground_path": "/path/to/mask.zarr/s0",
  "blocksize": [256, 256, 256],
  "model_kwargs": {
    "gpu": true,
    "pretrained_model": "/path/to/custom/model"
  },
  "eval_kwargs": {
    "diameter": 30,
    "z_axis": 0,
    "channels": [0, 0],
    "min_size": 1000,
    "anisotropy": 2.0,
    "do_3D": true
  },
  "cluster_kwargs": {
    "ncpus": 4,
    "min_workers": 2,
    "max_workers": 8,
    "queue": "normal",
    "memory_limit": "8GB"
  }
}
```

#### Running Segmentation

```python
from cellpose.utils import run_cellpose_distributed, get_foreground_mask

# Create foreground mask (optional)
get_foreground_mask(
    input_zarr_path='input.zarr/s0',
    output_zarr_path='mask.zarr',
    restrict=0.1,
    min_object_size=1000
)

# Run distributed segmentation
segments, boxes = run_cellpose_distributed('config.json')
```

### 2. Registration Quality Metrics

```python
from bigstream.metrics import ncc, mutual_information, metrics_sitk
import numpy as np

# Load your registered images
fixed = np.load('reference_image.npy')
moving = np.load('registered_image.npy')

# Compute metrics
correlation = ncc(fixed, moving)
mi_score = mutual_information(fixed, moving, bins=32)

# Use SimpleITK for additional metrics
sitk_metrics = metrics_sitk(fixed, moving)
print(f"Correlation: {sitk_metrics['correlation']}")
print(f"Mutual Information: {sitk_metrics['mutual_information']}")
print(f"Mean Squares Error: {sitk_metrics['mean_squares']}")
```

## API Reference

### Cellpose Module (`cellpose/utils.py`)

#### `get_foreground_mask(input_zarr_path, output_zarr_path, restrict=0.1, min_object_size=1000, closing_radius=2)`
Creates a 3D foreground mask using Otsu thresholding and morphological operations.

**Parameters:**
- `input_zarr_path`: Path to input zarr array
- `output_zarr_path`: Path for output mask
- `restrict`: Threshold restriction factor (0-1)
- `min_object_size`: Minimum object size to retain
- `closing_radius`: Radius for morphological closing

#### `run_cellpose_distributed(config_path)`
Runs distributed Cellpose segmentation based on JSON configuration.

**Parameters:**
- `config_path`: Path to JSON configuration file

**Returns:**
- `segments`: Zarr array with segmentation labels
- `boxes`: List of bounding boxes for detected objects

### Metrics Module (`bigstream/metrics.py`)

#### `ncc(fixed, moving, mask=None)`
Computes normalized cross-correlation between images.

**Parameters:**
- `fixed`: Reference image
- `moving`: Image to compare
- `mask`: Optional binary mask

**Returns:**
- NCC value (range: -1 to 1, higher is better)

#### `mutual_information(fixed, moving, bins=32, mask=None)`
Computes mutual information between images.

#### `metrics_sitk(fixed, moving)`
Computes comprehensive metrics using SimpleITK including correlation, mutual information, and mean squares error.

## Configuration Parameters

### Cellpose

### Model Parameters (`model_kwargs`)
- `gpu`: Enable GPU acceleration
- `pretrained_model`: Path to custom Cellpose model

### Evaluation Parameters (`eval_kwargs`)
- `diameter`: Expected cell diameter in pixels
- `z_axis`: Z-axis index (0, 1, or 2)
- `channels`: [cytoplasm_channel, nucleus_channel]
- `min_size`: Minimum object size
- `anisotropy`: Z-anisotropy factor
- `do_3D`: Enable 3D segmentation

### Cluster Parameters (`cluster_kwargs`)
For LSF clusters:
- `ncpus`: CPUs per worker
- `min_workers`/`max_workers`: Worker limits
- `queue`: LSF queue name
- `memory_limit`: Memory per worker





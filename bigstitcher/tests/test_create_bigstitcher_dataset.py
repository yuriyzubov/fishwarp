"""Test for create_bigstitcher_dataset function."""

import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

import dask.array as da
import numpy as np
import pytest
import zarr

from bigstitcher.to_bigstitcher import create_bigstitcher_dataset


def test_create_bigstitcher_dataset_with_numpy_arrays():
    """Test creating a BigStitcher dataset from numpy arrays."""
    # Create small test arrays (z, y, x)
    tile1 = np.random.randint(0, 255, size=(10, 32, 32), dtype=np.uint8)
    tile2 = np.random.randint(0, 255, size=(10, 32, 32), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = create_bigstitcher_dataset(
            zarr_arrays=[tile1, tile2],
            voxel_size=(0.5, 0.5, 1.0),
            output_folder=tmpdir,
            tile_names=["tile_0", "tile_1"],
            downsampling_factors=[(2, 2, 2)],  # Only one level to speed up test
            n_workers=1,
            threads_per_worker=1,
            memory_limit="1GB"
        )

        # Check output folder structure
        assert output_path == Path(tmpdir)
        assert (output_path / "dataset.xml").exists()
        assert (output_path / "dataset.zarr").exists()

        # Check XML is valid and has expected structure
        tree = ET.parse(output_path / "dataset.xml")
        root = tree.getroot()
        assert root.tag == "SpimData"

        # Check zarr has expected groups
        store = zarr.open(output_path / "dataset.zarr", mode='r')
        assert "s0-t0.zarr" in store
        assert "s1-t0.zarr" in store



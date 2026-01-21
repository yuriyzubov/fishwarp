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


def test_create_bigstitcher_dataset_with_dask_arrays():
    """Test creating a BigStitcher dataset from dask arrays."""
    tile1 = da.random.randint(0, 255, size=(10, 32, 32), dtype=np.uint8, chunks=(10, 32, 32))
    tile2 = da.random.randint(0, 255, size=(10, 32, 32), dtype=np.uint8, chunks=(10, 32, 32))

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = create_bigstitcher_dataset(
            zarr_arrays=[tile1, tile2],
            voxel_size=(0.259, 0.259, 1.0),
            output_folder=tmpdir,
            downsampling_factors=[(2, 2, 2)],
            n_workers=1,
            threads_per_worker=1,
            memory_limit="1GB"
        )

        assert (output_path / "dataset.xml").exists()
        assert (output_path / "dataset.zarr").exists()


def test_create_bigstitcher_dataset_with_zarr_arrays():
    """Test creating a BigStitcher dataset from zarr arrays."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create source zarr arrays
        source_path = Path(tmpdir) / "source"
        source_path.mkdir()

        tile1 = zarr.open(source_path / "tile1.zarr", mode='w', shape=(10, 32, 32), dtype=np.uint8)
        tile1[:] = np.random.randint(0, 255, size=(10, 32, 32), dtype=np.uint8)

        tile2 = zarr.open(source_path / "tile2.zarr", mode='w', shape=(10, 32, 32), dtype=np.uint8)
        tile2[:] = np.random.randint(0, 255, size=(10, 32, 32), dtype=np.uint8)

        output_folder = Path(tmpdir) / "output"
        output_path = create_bigstitcher_dataset(
            zarr_arrays=[tile1, tile2],
            voxel_size=(0.5, 0.5, 1.0),
            output_folder=output_folder,
            downsampling_factors=[(2, 2, 2)],
            n_workers=1,
            threads_per_worker=1,
            memory_limit="1GB"
        )

        assert (output_path / "dataset.xml").exists()
        assert (output_path / "dataset.zarr").exists()

def test_create_bigstitcher_dataset_multiscale_levels():
    """Test that multiple downsampling levels are created."""
    tile = np.random.randint(0, 255, size=(16, 64, 64), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = create_bigstitcher_dataset(
            zarr_arrays=[tile],
            voxel_size=(1.0, 1.0, 1.0),
            output_folder=tmpdir,
            downsampling_factors=[(2, 2, 2), (4, 4, 4)],
            n_workers=1,
            threads_per_worker=1,
            memory_limit="1GB"
        )

        store = zarr.open(output_path / "dataset.zarr", mode='r')
        view_group = store["s0-t0.zarr"]

        # Check all resolution levels exist (0=base, 1=2x, 2=4x)
        assert "0" in view_group
        assert "1" in view_group
        assert "2" in view_group


def test_create_bigstitcher_dataset_xml_content():
    """Test that XML contains correct metadata."""
    tile = np.random.randint(0, 255, size=(10, 32, 32), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = create_bigstitcher_dataset(
            zarr_arrays=[tile],
            voxel_size=(0.5, 0.5, 1.0),
            output_folder=tmpdir,
            tile_names=["my_tile"],
            channel_names=["GFP"],
            voxel_unit="micrometer",
            downsampling_factors=[(2, 2, 2)],
            n_workers=1,
            threads_per_worker=1,
            memory_limit="1GB"
        )

        tree = ET.parse(output_path / "dataset.xml")
        root = tree.getroot()

        # Check voxel size
        voxel_size = root.find(".//voxelSize/size")
        assert voxel_size.text == "0.5 0.5 1.0"

        # Check voxel unit
        voxel_unit = root.find(".//voxelSize/unit")
        assert voxel_unit.text == "micrometer"

        # Check tile name
        tile_name = root.find(".//Attributes[@name='tile']/Tile/name")
        assert tile_name.text == "my_tile"

        # Check channel name
        channel_name = root.find(".//Attributes[@name='channel']/Channel/name")
        assert channel_name.text == "GFP"

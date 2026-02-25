"""Test for create_bigstitcher_dataset function."""

import tempfile
import warnings
from pathlib import Path
from xml.etree import ElementTree as ET

import dask.array as da
import numpy as np
import pytest
import zarr

from bigstitcher.to_bigstitcher import (
    _parse_interest_points_n5,
    _read_base_shape,
    _write_dataset_xml,
    add_interest_points_to_xml,
    create_bigstitcher_dataset,
    create_bigstitcher_dataset_symlinked,
)


# ─── shared test data ────────────────────────────────────────────────────────

_VIEW_SETUPS = [
    {
        'id': 0,
        'name': 's0-t0',
        'size': (32, 32, 10),
        'voxel_size': (0.5, 0.5, 1.0),
        'tile_id': 0,
        'tile_name': 'tile_0',
        'channel_id': 0,
        'timepoint': 0,
    }
]
_ZGROUPS = [{'setup': 0, 'tp': 0, 'path': 's0-t0.zarr', 'indices': '0 0'}]


def _make_n5_interest_points(path: Path, entries) -> Path:
    """Create a minimal interestpoints.n5 store.

    entries: list of (timepoint, setup, label) tuples.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        store = zarr.N5Store(str(path))
    root = zarr.open_group(store=store, mode='w')
    for tp, setup, label in entries:
        root.require_group(f"tpId_{tp}_viewSetupId_{setup}/{label}")
    return path


def _make_source_zarr(path: Path, shape=(10, 32, 32)) -> Path:
    """Create a minimal zarr group with a '0' resolution level."""
    grp = zarr.open_group(str(path), mode='w')
    grp.create_dataset("0", data=np.zeros(shape, dtype=np.uint8))
    return path


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

        # <ViewInterestPoints> is present but empty when no interest points given
        vip = root.find("ViewInterestPoints")
        assert vip is not None
        assert list(vip) == []


# ─── _read_base_shape ────────────────────────────────────────────────────────

def test_read_base_shape_from_multiscales_metadata():
    """Reads shape from OME-Zarr multiscales metadata (preferred path)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        grp = zarr.open_group(tmpdir, mode='w')
        grp.create_dataset("0", data=np.zeros((10, 32, 32), dtype=np.uint8))
        grp.attrs["multiscales"] = [{"datasets": [{"path": "0"}]}]

        shape = _read_base_shape(grp)
        assert shape == (10, 32, 32)


def test_read_base_shape_fallback_to_level_0():
    """Falls back to the '0' subarray when no multiscales metadata exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        grp = zarr.open_group(tmpdir, mode='w')
        grp.create_dataset("0", data=np.zeros((5, 16, 16), dtype=np.uint8))

        shape = _read_base_shape(grp)
        assert shape == (5, 16, 16)


def test_read_base_shape_fallback_to_s0():
    """Falls back to the 's0' subarray when '0' is absent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        grp = zarr.open_group(tmpdir, mode='w')
        grp.create_dataset("s0", data=np.zeros((8, 64, 64), dtype=np.uint8))

        shape = _read_base_shape(grp)
        assert shape == (8, 64, 64)


def test_read_base_shape_raises_on_unrecognized_structure():
    """Raises ValueError when the group has no recognizable shape source."""
    with tempfile.TemporaryDirectory() as tmpdir:
        grp = zarr.open_group(tmpdir, mode='w')
        # Only an unrecognized subgroup, no metadata
        grp.require_group("raw_data")

        with pytest.raises(ValueError, match="Cannot determine shape"):
            _read_base_shape(grp)


# ─── _parse_interest_points_n5 ───────────────────────────────────────────────

def test_parse_interest_points_n5_missing_path_returns_empty():
    """Returns an empty list when the n5 path does not exist."""
    entries = _parse_interest_points_n5("/nonexistent/path/interestpoints.n5")
    assert entries == []


def test_parse_interest_points_n5_single_entry():
    """Parses a single timepoint/setup/label correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        n5_path = Path(tmpdir) / "interestpoints.n5"
        _make_n5_interest_points(n5_path, [(0, 0, "beads")])

        entries = _parse_interest_points_n5(n5_path)

        assert len(entries) == 1
        assert entries[0]["timepoint"] == 0
        assert entries[0]["setup"] == 0
        assert entries[0]["label"] == "beads"
        assert entries[0]["path"] == "tpId_0_viewSetupId_0/beads"


def test_parse_interest_points_n5_multiple_setups_and_labels():
    """Parses multiple setups and labels, returning one entry per combination."""
    with tempfile.TemporaryDirectory() as tmpdir:
        n5_path = Path(tmpdir) / "interestpoints.n5"
        _make_n5_interest_points(n5_path, [
            (0, 0, "beads"),
            (0, 0, "blobs"),
            (0, 1, "beads"),
        ])

        entries = _parse_interest_points_n5(n5_path)

        assert len(entries) == 3
        paths = {e["path"] for e in entries}
        assert "tpId_0_viewSetupId_0/beads" in paths
        assert "tpId_0_viewSetupId_0/blobs" in paths
        assert "tpId_0_viewSetupId_1/beads" in paths


def test_parse_interest_points_n5_ignores_unexpected_group_names():
    """Groups not matching tpId_N_viewSetupId_M are silently skipped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        n5_path = Path(tmpdir) / "interestpoints.n5"
        _make_n5_interest_points(n5_path, [(0, 0, "beads")])
        # Add an extra group with an unrecognized name
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            store = zarr.N5Store(str(n5_path))
        zarr.open_group(store=store, mode='a').require_group("attributes")

        entries = _parse_interest_points_n5(n5_path)

        assert len(entries) == 1
        assert entries[0]["setup"] == 0


# ─── _write_dataset_xml ──────────────────────────────────────────────────────

def test_write_dataset_xml_interest_points_none_produces_empty_element():
    """<ViewInterestPoints> is present but empty when interest_points=None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "dataset.xml"
        _write_dataset_xml(xml_path, _VIEW_SETUPS, _ZGROUPS, "micrometer", ["ch0"])

        root = ET.parse(xml_path).getroot()
        vip = root.find("ViewInterestPoints")
        assert vip is not None
        assert list(vip) == []


def test_write_dataset_xml_interest_points_empty_list_produces_empty_element():
    """<ViewInterestPoints> is empty when interest_points=[]."""
    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "dataset.xml"
        _write_dataset_xml(xml_path, _VIEW_SETUPS, _ZGROUPS, "micrometer", ["ch0"],
                           interest_points=[])

        root = ET.parse(xml_path).getroot()
        vip = root.find("ViewInterestPoints")
        assert vip is not None
        assert list(vip) == []


def test_write_dataset_xml_interest_points_written_correctly():
    """Each interest point entry becomes a <ViewInterestPointsFile> with correct attributes."""
    ip_entries = [
        {"timepoint": 0, "setup": 0, "label": "beads",
         "path": "tpId_0_viewSetupId_0/beads", "params": "manual"},
        {"timepoint": 0, "setup": 1, "label": "beads",
         "path": "tpId_0_viewSetupId_1/beads"},  # no 'params' key → default
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "dataset.xml"
        _write_dataset_xml(xml_path, _VIEW_SETUPS, _ZGROUPS, "micrometer", ["ch0"],
                           interest_points=ip_entries)

        root = ET.parse(xml_path).getroot()
        vip_files = root.findall("ViewInterestPoints/ViewInterestPointsFile")
        assert len(vip_files) == 2

        first = vip_files[0]
        assert first.get("timepoint") == "0"
        assert first.get("setup") == "0"
        assert first.get("label") == "beads"
        assert first.get("params") == "manual"
        assert first.text == "tpId_0_viewSetupId_0/beads"

        # Missing 'params' key defaults to "manual"
        second = vip_files[1]
        assert second.get("params") == "manual"

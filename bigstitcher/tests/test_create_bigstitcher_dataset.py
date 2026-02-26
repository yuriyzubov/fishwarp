"""Tests for create_bigstitcher_dataset and related helpers."""

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

# ViewSetups are unique per (tile, channel) — no 'timepoint' key.
_VIEW_SETUPS = [
    {
        'id': 0,
        'name': 'tile_0',
        'size': (32, 32, 10),
        'voxel_size': (0.5, 0.5, 1.0),
        'tile_id': 0,
        'tile_name': 'tile_0',
        'channel_id': 0,
    }
]
_ZGROUPS = [{'setup': 0, 'tp': 0, 'path': 'tile_0.zarr', 'indices': '0 0'}]
_VOXEL_SIZE = (0.5, 0.5, 1.0)


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


# ─── create_bigstitcher_dataset ───────────────────────────────────────────────

def test_create_bigstitcher_dataset_with_numpy_arrays():
    """Test creating a BigStitcher dataset from numpy arrays."""
    tile1 = np.random.randint(0, 255, size=(10, 32, 32), dtype=np.uint8)
    tile2 = np.random.randint(0, 255, size=(10, 32, 32), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = create_bigstitcher_dataset(
            zarr_arrays=[tile1, tile2],
            voxel_size=(0.5, 0.5, 1.0),
            output_folder=tmpdir,
            tile_names=["tile_0", "tile_1"],
            downsampling_factors=[(2, 2, 2)],
            n_workers=1,
            threads_per_worker=1,
            memory_limit="1GB",
        )

        assert output_path == Path(tmpdir)
        assert (output_path / "dataset.xml").exists()
        assert (output_path / "dataset.zarr").exists()

        tree = ET.parse(output_path / "dataset.xml")
        root = tree.getroot()
        assert root.tag == "SpimData"

        store = zarr.open(output_path / "dataset.zarr", mode='r')
        assert "tile_0.zarr" in store
        assert "tile_1.zarr" in store


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
            memory_limit="1GB",
        )

        assert (output_path / "dataset.xml").exists()
        assert (output_path / "dataset.zarr").exists()


def test_create_bigstitcher_dataset_with_zarr_arrays():
    """Test creating a BigStitcher dataset from zarr arrays."""
    with tempfile.TemporaryDirectory() as tmpdir:
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
            memory_limit="1GB",
        )

        assert (output_path / "dataset.xml").exists()
        assert (output_path / "dataset.zarr").exists()


def test_create_bigstitcher_dataset_multiscale_levels():
    """Multiple downsampling levels are created for each tile."""
    tile = np.random.randint(0, 255, size=(16, 64, 64), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = create_bigstitcher_dataset(
            zarr_arrays=[tile],
            voxel_size=(1.0, 1.0, 1.0),
            output_folder=tmpdir,
            downsampling_factors=[(2, 2, 2), (4, 4, 4)],
            n_workers=1,
            threads_per_worker=1,
            memory_limit="1GB",
        )

        store = zarr.open(output_path / "dataset.zarr", mode='r')
        view_group = store["tile_0.zarr"]

        # 0 = base, 1 = 2x, 2 = 4x
        assert "0" in view_group
        assert "1" in view_group
        assert "2" in view_group


def test_create_bigstitcher_dataset_xml_content():
    """XML contains correct voxel size, tile name, channel name, and empty ViewInterestPoints."""
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
            memory_limit="1GB",
        )

        root = ET.parse(output_path / "dataset.xml").getroot()

        assert root.find(".//voxelSize/size").text == "0.5 0.5 1.0"
        assert root.find(".//voxelSize/unit").text == "micrometer"
        assert root.find(".//Attributes[@name='tile']/Tile/name").text == "my_tile"
        assert root.find(".//Attributes[@name='channel']/Channel/name").text == "GFP"

        vip = root.find("ViewInterestPoints")
        assert vip is not None
        assert list(vip) == []


def test_create_bigstitcher_dataset_3d_native():
    """3D input produces native 3D zarr arrays and indicies='[]' in the XML."""
    tile = np.random.randint(0, 255, size=(10, 32, 32), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = create_bigstitcher_dataset(
            zarr_arrays=[tile],
            voxel_size=(0.5, 0.5, 1.0),
            output_folder=tmpdir,
            downsampling_factors=[(2, 2, 2)],
            n_workers=1,
            threads_per_worker=1,
            memory_limit="1GB",
        )

        # Verify indicies="[]" in the XML (3D native, no TCZYX indexing)
        root = ET.parse(output_path / "dataset.xml").getroot()
        zgroup = root.find(".//zgroup")
        assert zgroup is not None
        assert zgroup.get("indicies") == "[]"

        # Verify the zarr level 0 is written as 3D (z, y, x), not 5D
        store = zarr.open(output_path / "dataset.zarr", mode='r')
        level0 = store["tile_0.zarr"]["0"]
        assert level0.ndim == 3


def test_create_bigstitcher_dataset_multichannel():
    """4D input (c, z, y, x) creates one ViewSetup per channel."""
    # 2 channels, z=10, y=32, x=32
    tile = np.random.randint(0, 255, size=(2, 10, 32, 32), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = create_bigstitcher_dataset(
            zarr_arrays=[tile],
            voxel_size=(0.5, 0.5, 1.0),
            output_folder=tmpdir,
            channel_names=["DAPI", "GFP"],
            downsampling_factors=[(2, 2, 2)],
            n_workers=1,
            threads_per_worker=1,
            memory_limit="1GB",
        )

        root = ET.parse(output_path / "dataset.xml").getroot()

        # One ViewSetup per channel
        setups = root.findall(".//ViewSetup")
        assert len(setups) == 2

        # Two channels declared
        channels = root.findall(".//Attributes[@name='channel']/Channel")
        assert len(channels) == 2
        names = {ch.find("name").text for ch in channels}
        assert names == {"DAPI", "GFP"}

        # indicies encode channel index ("0 0" and "0 1")
        zgroups = root.findall(".//zgroup")
        indices_vals = {zg.get("indicies") for zg in zgroups}
        assert "0 0" in indices_vals
        assert "0 1" in indices_vals

        # Data written as 5D zarr
        store = zarr.open(output_path / "dataset.zarr", mode='r')
        level0 = store["tile_0.zarr"]["0"]
        assert level0.ndim == 5


def test_create_bigstitcher_dataset_calibration_affine():
    """ViewRegistration calibration affine encodes voxel_size on the diagonal."""
    tile = np.zeros((10, 32, 32), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = create_bigstitcher_dataset(
            zarr_arrays=[tile],
            voxel_size=(0.259, 0.259, 1.0),
            output_folder=tmpdir,
            downsampling_factors=[(2, 2, 2)],
            n_workers=1,
            threads_per_worker=1,
            memory_limit="1GB",
        )

        root = ET.parse(output_path / "dataset.xml").getroot()
        affine = root.find(".//ViewTransform[@type='affine']/affine")
        assert affine is not None
        assert affine.text == "0.259 0.0 0.0 0.0 0.0 0.259 0.0 0.0 0.0 0.0 1.0 0.0"


def test_create_bigstitcher_dataset_mismatched_ndim_raises():
    """Mixing input arrays with different numbers of dimensions raises ValueError."""
    arr3d = np.zeros((10, 32, 32), dtype=np.uint8)
    arr5d = np.zeros((1, 1, 10, 32, 32), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="dimensions"):
            create_bigstitcher_dataset(
                zarr_arrays=[arr3d, arr5d],
                voxel_size=(1.0, 1.0, 1.0),
                output_folder=tmpdir,
                downsampling_factors=[(2, 2, 2)],
                n_workers=1,
                threads_per_worker=1,
                memory_limit="1GB",
            )


# ─── _read_base_shape ────────────────────────────────────────────────────────

def test_read_base_shape_from_multiscales_metadata():
    """Reads shape from OME-Zarr multiscales metadata (preferred path)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        grp = zarr.open_group(tmpdir, mode='w')
        grp.create_dataset("0", data=np.zeros((10, 32, 32), dtype=np.uint8))
        grp.attrs["multiscales"] = [{"datasets": [{"path": "0"}]}]

        assert _read_base_shape(grp) == (10, 32, 32)


def test_read_base_shape_fallback_to_level_0():
    """Falls back to the '0' subarray when no multiscales metadata exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        grp = zarr.open_group(tmpdir, mode='w')
        grp.create_dataset("0", data=np.zeros((5, 16, 16), dtype=np.uint8))

        assert _read_base_shape(grp) == (5, 16, 16)


def test_read_base_shape_fallback_to_s0():
    """Falls back to the 's0' subarray when '0' is absent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        grp = zarr.open_group(tmpdir, mode='w')
        grp.create_dataset("s0", data=np.zeros((8, 64, 64), dtype=np.uint8))

        assert _read_base_shape(grp) == (8, 64, 64)


def test_read_base_shape_raises_on_unrecognized_structure():
    """Raises ValueError when the group has no recognizable shape source."""
    with tempfile.TemporaryDirectory() as tmpdir:
        grp = zarr.open_group(tmpdir, mode='w')
        grp.require_group("raw_data")

        with pytest.raises(ValueError, match="Cannot determine shape"):
            _read_base_shape(grp)


# ─── _parse_interest_points_n5 ───────────────────────────────────────────────

def test_parse_interest_points_n5_missing_path_returns_empty():
    """Returns an empty list when the n5 path does not exist."""
    assert _parse_interest_points_n5("/nonexistent/path/interestpoints.n5") == []


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
        _write_dataset_xml(xml_path, _VIEW_SETUPS, _ZGROUPS, "micrometer",
                           _VOXEL_SIZE, ["ch0"])

        vip = ET.parse(xml_path).getroot().find("ViewInterestPoints")
        assert vip is not None
        assert list(vip) == []


def test_write_dataset_xml_interest_points_empty_list_produces_empty_element():
    """<ViewInterestPoints> is empty when interest_points=[]."""
    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "dataset.xml"
        _write_dataset_xml(xml_path, _VIEW_SETUPS, _ZGROUPS, "micrometer",
                           _VOXEL_SIZE, ["ch0"], interest_points=[])

        vip = ET.parse(xml_path).getroot().find("ViewInterestPoints")
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
        _write_dataset_xml(xml_path, _VIEW_SETUPS, _ZGROUPS, "micrometer",
                           _VOXEL_SIZE, ["ch0"], interest_points=ip_entries)

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
        assert vip_files[1].get("params") == "manual"


# ─── add_interest_points_to_xml ──────────────────────────────────────────────

def test_add_interest_points_to_xml_populates_view_interest_points():
    """Writes interest point entries into an existing XML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "dataset.xml"
        n5_path = Path(tmpdir) / "interestpoints.n5"
        _write_dataset_xml(xml_path, _VIEW_SETUPS, _ZGROUPS, "micrometer",
                           _VOXEL_SIZE, ["ch0"])
        _make_n5_interest_points(n5_path, [(0, 0, "beads")])

        add_interest_points_to_xml(xml_path, n5_path)

        vip_files = ET.parse(xml_path).getroot().findall(
            "ViewInterestPoints/ViewInterestPointsFile"
        )
        assert len(vip_files) == 1
        assert vip_files[0].get("label") == "beads"
        assert vip_files[0].get("timepoint") == "0"
        assert vip_files[0].get("setup") == "0"


def test_add_interest_points_to_xml_replaces_existing_entries():
    """Calling the function twice replaces, not appends, the entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "dataset.xml"
        n5_first = Path(tmpdir) / "ip_first.n5"
        n5_second = Path(tmpdir) / "ip_second.n5"
        _write_dataset_xml(xml_path, _VIEW_SETUPS, _ZGROUPS, "micrometer",
                           _VOXEL_SIZE, ["ch0"])
        _make_n5_interest_points(n5_first, [(0, 0, "blobs")])
        _make_n5_interest_points(n5_second, [(0, 0, "beads"), (0, 1, "beads")])

        add_interest_points_to_xml(xml_path, n5_first)
        add_interest_points_to_xml(xml_path, n5_second)

        vip_files = ET.parse(xml_path).getroot().findall(
            "ViewInterestPoints/ViewInterestPointsFile"
        )
        assert len(vip_files) == 2
        assert {el.get("label") for el in vip_files} == {"beads"}


def test_add_interest_points_to_xml_raises_if_element_missing():
    """Raises ValueError when the XML has no <ViewInterestPoints> element."""
    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "broken.xml"
        xml_path.write_text('<?xml version="1.0" ?><SpimData version="0.2"></SpimData>')
        n5_path = Path(tmpdir) / "interestpoints.n5"
        _make_n5_interest_points(n5_path, [(0, 0, "beads")])

        with pytest.raises(ValueError, match="ViewInterestPoints"):
            add_interest_points_to_xml(xml_path, n5_path)


def test_add_interest_points_to_xml_empty_when_n5_missing():
    """When the n5 path does not exist, <ViewInterestPoints> is left empty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "dataset.xml"
        _write_dataset_xml(xml_path, _VIEW_SETUPS, _ZGROUPS, "micrometer",
                           _VOXEL_SIZE, ["ch0"])

        add_interest_points_to_xml(xml_path, Path(tmpdir) / "nonexistent.n5")

        assert list(ET.parse(xml_path).getroot().find("ViewInterestPoints")) == []


# ─── create_bigstitcher_dataset_symlinked ────────────────────────────────────

def test_create_bigstitcher_dataset_symlinked_creates_symlinks():
    """Symlinks point to source zarr paths; no data is copied."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src0 = _make_source_zarr(Path(tmpdir) / "src0.zarr")
        src1 = _make_source_zarr(Path(tmpdir) / "src1.zarr")
        out = Path(tmpdir) / "out"

        create_bigstitcher_dataset_symlinked(
            zarr_paths=[src0, src1],
            voxel_size=(0.5, 0.5, 1.0),
            output_folder=out,
        )

        link0 = out / "dataset.zarr" / "tile_0.zarr"
        link1 = out / "dataset.zarr" / "tile_1.zarr"
        assert link0.is_symlink()
        assert link1.is_symlink()
        assert link0.resolve() == src0.resolve()
        assert link1.resolve() == src1.resolve()


def test_create_bigstitcher_dataset_symlinked_replaces_existing_symlink():
    """A stale symlink at the target path is replaced without error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = _make_source_zarr(Path(tmpdir) / "src.zarr")
        out = Path(tmpdir) / "out"
        (out / "dataset.zarr").mkdir(parents=True)
        stale_link = out / "dataset.zarr" / "tile_0.zarr"
        stale_link.symlink_to("/nonexistent/stale")

        create_bigstitcher_dataset_symlinked(
            zarr_paths=[src],
            voxel_size=(0.5, 0.5, 1.0),
            output_folder=out,
        )

        assert stale_link.resolve() == src.resolve()


def test_create_bigstitcher_dataset_symlinked_xml_content():
    """XML contains the correct voxel size, tile name, and array size."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = _make_source_zarr(Path(tmpdir) / "src.zarr", shape=(10, 32, 48))
        out = Path(tmpdir) / "out"

        create_bigstitcher_dataset_symlinked(
            zarr_paths=[src],
            voxel_size=(0.25, 0.25, 1.0),
            output_folder=out,
            tile_names=["my_tile"],
            channel_names=["DAPI"],
            voxel_unit="micrometer",
        )

        root = ET.parse(out / "dataset.xml").getroot()

        assert root.find(".//voxelSize/size").text == "0.25 0.25 1.0"
        assert root.find(".//voxelSize/unit").text == "micrometer"
        assert root.find(".//Attributes[@name='tile']/Tile/name").text == "my_tile"
        assert root.find(".//Attributes[@name='channel']/Channel/name").text == "DAPI"
        # shape (10, 32, 48) → z=10, y=32, x=48 → size "48 32 10"
        assert root.find(".//ViewSetup/size").text == "48 32 10"


def test_create_bigstitcher_dataset_symlinked_default_names():
    """Default tile and channel names are generated when not provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src0 = _make_source_zarr(Path(tmpdir) / "src0.zarr")
        src1 = _make_source_zarr(Path(tmpdir) / "src1.zarr")
        out = Path(tmpdir) / "out"

        create_bigstitcher_dataset_symlinked(
            zarr_paths=[src0, src1],
            voxel_size=(1.0, 1.0, 1.0),
            output_folder=out,
        )

        root = ET.parse(out / "dataset.xml").getroot()
        tile_names = [
            el.text for el in root.findall(".//Attributes[@name='tile']/Tile/name")
        ]
        assert tile_names == ["tile_0", "tile_1"]
        assert root.find(".//Attributes[@name='channel']/Channel/name").text == "channel_0"


def test_create_bigstitcher_dataset_symlinked_with_interest_points():
    """<ViewInterestPoints> is populated when interest_points_n5 is given."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = _make_source_zarr(Path(tmpdir) / "src.zarr")
        n5_path = _make_n5_interest_points(
            Path(tmpdir) / "interestpoints.n5", [(0, 0, "beads")]
        )
        out = Path(tmpdir) / "out"

        create_bigstitcher_dataset_symlinked(
            zarr_paths=[src],
            voxel_size=(1.0, 1.0, 1.0),
            output_folder=out,
            interest_points_n5=n5_path,
        )

        vip_files = ET.parse(out / "dataset.xml").getroot().findall(
            "ViewInterestPoints/ViewInterestPointsFile"
        )
        assert len(vip_files) == 1
        assert vip_files[0].get("label") == "beads"


def test_create_bigstitcher_dataset_symlinked_without_interest_points():
    """<ViewInterestPoints> is empty when interest_points_n5 is not provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = _make_source_zarr(Path(tmpdir) / "src.zarr")
        out = Path(tmpdir) / "out"

        create_bigstitcher_dataset_symlinked(
            zarr_paths=[src],
            voxel_size=(1.0, 1.0, 1.0),
            output_folder=out,
        )

        assert list(
            ET.parse(out / "dataset.xml").getroot().find("ViewInterestPoints")
        ) == []


# ─── create_bigstitcher_dataset — interest_points_n5 ─────────────────────────

def test_create_bigstitcher_dataset_with_interest_points_n5():
    """<ViewInterestPoints> is populated when interest_points_n5 is passed."""
    tile = np.random.randint(0, 255, size=(10, 32, 32), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        n5_path = _make_n5_interest_points(
            Path(tmpdir) / "interestpoints.n5",
            [(0, 0, "beads"), (0, 0, "blobs")],
        )

        output_path = create_bigstitcher_dataset(
            zarr_arrays=[tile],
            voxel_size=(0.5, 0.5, 1.0),
            output_folder=tmpdir,
            downsampling_factors=[(2, 2, 2)],
            n_workers=1,
            threads_per_worker=1,
            memory_limit="1GB",
            interest_points_n5=n5_path,
        )

        vip_files = ET.parse(output_path / "dataset.xml").getroot().findall(
            "ViewInterestPoints/ViewInterestPointsFile"
        )
        assert len(vip_files) == 2
        assert {el.get("label") for el in vip_files} == {"beads", "blobs"}

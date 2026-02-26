"""
BigStitcher Zarr Dataset Creator

This module provides functionality to convert zarr arrays into a BigStitcher-compatible
dataset with proper XML metadata for opening in BigStitcher/BigDataViewer.

Uses Dask for lazy loading and parallel data copying with LocalCluster.
"""

import os
import json
import numpy as np
import zarr
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster
from pathlib import Path
from typing import List, Tuple, Union, Optional
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom


def create_bigstitcher_dataset(
    zarr_arrays: List[Union[zarr.Array, np.ndarray, da.Array, str, Path]],
    voxel_size: Tuple[float, float, float],
    output_folder: Union[str, Path],
    voxel_unit: str = "micrometer",
    tile_names: Optional[List[str]] = None,
    channel_names: Optional[List[str]] = None,
    downsampling_factors: Optional[List[Tuple[int, int, int]]] = None,
    chunk_size: Tuple[int, int, int] = (64, 128, 128),
    compression: str = "zstd",
    compression_level: int = 3,
    n_workers: int = 4,
    threads_per_worker: int = 2,
    memory_limit: str = "4GB",
    interest_points_n5: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Convert zarr arrays into a BigStitcher-compatible dataset using Dask for
    lazy loading and parallel processing.

    Parameters
    ----------
    zarr_arrays : List[Union[zarr.Array, np.ndarray, da.Array, str, Path]]
        List of input arrays. Can be:
        - zarr.Array objects
        - numpy arrays
        - dask arrays
        - Paths to existing zarr arrays
        Each array should be 3D (z, y, x), 4D (c, z, y, x), or 5D (t, c, z, y, x)

    voxel_size : Tuple[float, float, float]
        Voxel size in (x, y, z) order, in the units specified by voxel_unit.

    output_folder : Union[str, Path]
        Destination folder for the BigStitcher dataset.
        Will create dataset.zarr/ and dataset.xml inside this folder.

    voxel_unit : str, optional
        Unit for voxel size. Default is "micrometer".
        Common values: "micrometer", "nanometer", "millimeter"

    tile_names : Optional[List[str]], optional
        Names for each tile/view. If None, uses "tile_0", "tile_1", etc.

    channel_names : Optional[List[str]], optional
        Names for each channel. If None, uses "channel_0", "channel_1", etc.

    downsampling_factors : Optional[List[Tuple[int, int, int]]], optional
        List of (z, y, x) downsampling factors for multi-resolution pyramid.
        Default is [(2, 2, 2), (4, 4, 4), (8, 8, 8), (16, 16, 16)]

    chunk_size : Tuple[int, int, int], optional
        Chunk size for zarr arrays in (z, y, x) order. Default is (64, 128, 128).

    compression : str, optional
        Compression algorithm. Default is "zstd". Options: "zstd", "gzip", "lz4", None

    compression_level : int, optional
        Compression level. Default is 3.

    n_workers : int, optional
        Number of Dask workers for parallel processing. Default is 4.

    threads_per_worker : int, optional
        Number of threads per Dask worker. Default is 2.

    memory_limit : str, optional
        Memory limit per worker. Default is "4GB".

    interest_points_n5 : str or Path, optional
        Path to an existing interestpoints.n5 directory. If provided, its group
        structure is parsed to discover timepoints, setups, and labels, and the
        corresponding <ViewInterestPointsFile> entries are written into the XML.
        If None (default), <ViewInterestPoints> is left empty.

    Returns
    -------
    Path
        Path to the created dataset folder containing dataset.xml and dataset.zarr

    Examples
    --------
    >>> import zarr
    >>> import dask.array as da
    >>>
    >>> # Open existing zarr arrays lazily
    >>> tile1 = zarr.open("path/to/tile1.zarr", mode='r')
    >>> tile2 = zarr.open("path/to/tile2.zarr", mode='r')
    >>>
    >>> # Convert to BigStitcher format with parallel processing
    >>> create_bigstitcher_dataset(
    ...     zarr_arrays=[tile1, tile2],
    ...     voxel_size=(0.5, 0.5, 1.0),  # x, y, z in micrometers
    ...     output_folder="./my_dataset",
    ...     tile_names=["left_tile", "right_tile"],
    ...     n_workers=8
    ... )
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    dataset_zarr_path = output_folder / "dataset.zarr"
    dataset_xml_path = output_folder / "dataset.xml"

    # Default downsampling factors if not provided
    if downsampling_factors is None:
        downsampling_factors = [(2, 2, 2), (4, 4, 4), (8, 8, 8), (16, 16, 16)]

    # Start Dask LocalCluster
    print(f"Starting Dask LocalCluster with {n_workers} workers...")
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit
    )
    client = Client(cluster)
    print(f"Dask dashboard available at: {client.dashboard_link}")

    try:
        # Load and normalize arrays as dask arrays (lazy)
        arrays_info = []
        for i, arr in enumerate(zarr_arrays):
            dask_arr, shape, dtype = _load_array_lazy(arr, chunk_size)

            # Normalize to 5D (t, c, z, y, x)
            normalized_shape, original_ndim = _normalize_shape(shape)

            arrays_info.append({
                'data': dask_arr,
                'shape': normalized_shape,
                'original_shape': shape,
                'original_ndim': original_ndim,
                'dtype': dtype,
                'index': i
            })

        # Generate tile names if not provided
        if tile_names is None:
            tile_names = [f"tile_{i}" for i in range(len(zarr_arrays))]

        # Determine number of channels from first array
        n_channels = arrays_info[0]['shape'][1]
        if channel_names is None:
            channel_names = [f"channel_{i}" for i in range(n_channels)]

        # Create the zarr store
        store = zarr.DirectoryStore(str(dataset_zarr_path))
        root = zarr.group(store=store, overwrite=True)

        # Setup info for XML generation
        view_setups = []
        zgroups = []

        # Get compressor
        compressor = _get_compressor(compression, compression_level)

        # Process each array as a separate view/tile
        for arr_info in arrays_info:
            tile_idx = arr_info['index']
            tile_name = tile_names[tile_idx]
            t, c, z, y, x = arr_info['shape']

            print(f"\nProcessing tile {tile_idx}: {tile_name} (shape: {arr_info['shape']})")

            # For simplicity, we create one ViewSetup per tile
            for tp in range(t):
                setup_id = tile_idx
                group_name = f"s{setup_id}-t{tp}.zarr"

                # Create the zarr group for this view
                view_group = root.create_group(group_name)

                # Get normalized dask array for this timepoint
                dask_data = _normalize_dask_array(
                    arr_info['data'],
                    arr_info['original_ndim'],
                    tp,
                    chunk_size
                )

                # Write base resolution (level 0)
                print(f"  Writing level 0 (full resolution)...")
                _write_resolution_level_dask(
                    view_group=view_group,
                    level=0,
                    dask_data=dask_data,
                    chunk_size=chunk_size,
                    compressor=compressor,
                    downsampling_factor=(1, 1, 1)
                )

                # Write downsampled levels, each cascading from the previous level.
                # ds_factor is absolute (relative to level 0), so compute the per-step
                # relative factor as the ratio between consecutive absolute factors.
                # After writing each level we reload it from the on-disk zarr array so
                # that the next level's dask graph starts from disk data rather than
                # chaining through all prior lazy computations.  Without this, computing
                # level N would silently re-execute all N-1 preceding downsampling passes
                # for every output chunk, causing memory to grow with pyramid depth.
                current_data = dask_data
                prev_factor = (1, 1, 1)
                for level_idx, ds_factor in enumerate(downsampling_factors, start=1):
                    rel_factor = tuple(ds_factor[i] // prev_factor[i] for i in range(3))
                    print(f"  Writing level {level_idx} (downsample {rel_factor} from level {level_idx - 1})...")
                    downsampled = _downsample_dask_array(current_data, rel_factor)
                    prev_factor = ds_factor
                    written_arr = _write_resolution_level_dask(
                        view_group=view_group,
                        level=level_idx,
                        dask_data=downsampled,
                        chunk_size=chunk_size,
                        compressor=compressor,
                        downsampling_factor=ds_factor
                    )
                    # Break the computation chain: read the written level back from disk.
                    current_data = da.from_zarr(written_arr)

                # Write multiscale metadata
                _write_multiscale_metadata(
                    view_group=view_group,
                    base_shape=(z, y, x),
                    downsampling_factors=downsampling_factors,
                    voxel_size=voxel_size,
                    voxel_unit=voxel_unit
                )

                # Collect info for XML
                view_setups.append({
                    'id': setup_id,
                    'name': f"s{setup_id}-t{tp}",
                    'size': (x, y, z),  # BigStitcher uses x, y, z order
                    'voxel_size': voxel_size,
                    'tile_id': tile_idx,
                    'tile_name': tile_name,
                    'channel_id': 0,
                    'timepoint': tp
                })

                zgroups.append({
                    'setup': setup_id,
                    'tp': tp,
                    'path': group_name,
                    'indices': "0 0"  # Always 0 0 since each zgroup contains data at index [0,0]
                })

        # Parse interest points N5 if provided
        ip_entries = []
        if interest_points_n5 is not None:
            ip_entries = _parse_interest_points_n5(interest_points_n5)
            print(f"\nFound {len(ip_entries)} interest point entries in {interest_points_n5}")

        # Generate the XML file
        _write_dataset_xml(
            xml_path=dataset_xml_path,
            view_setups=view_setups,
            zgroups=zgroups,
            voxel_unit=voxel_unit,
            channel_names=channel_names,
            interest_points=ip_entries,
        )

        print(f"\nBigStitcher dataset created at: {output_folder}")
        print(f"  - dataset.zarr: {dataset_zarr_path}")
        print(f"  - dataset.xml: {dataset_xml_path}")

    finally:
        # Clean up Dask cluster
        print("\nShutting down Dask cluster...")
        client.close()
        cluster.close()

    return output_folder


def create_bigstitcher_dataset_symlinked(
    zarr_paths: List[Union[str, Path]],
    voxel_size: Tuple[float, float, float],
    output_folder: Union[str, Path],
    voxel_unit: str = "micrometer",
    tile_names: Optional[List[str]] = None,
    channel_names: Optional[List[str]] = None,
    interest_points_n5: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Create a BigStitcher-compatible dataset by symlinking existing zarr arrays.

    No data is copied. Each source zarr is symlinked into dataset.zarr/ and the
    XML is generated from the source metadata. Useful for large datasets where
    copying is impractical.

    The source zarr arrays must already have a compatible multiscale structure
    (OME-Zarr .zattrs with "multiscales", with resolution levels 0, 1, 2, ...).

    Parameters
    ----------
    zarr_paths : List[str or Path]
        Paths to existing zarr array groups. Each becomes one tile/view setup.

    voxel_size : Tuple[float, float, float]
        Voxel size in (x, y, z) order, in the units specified by voxel_unit.

    output_folder : str or Path
        Destination folder. Will contain dataset.xml and dataset.zarr/ (with symlinks).

    voxel_unit : str, optional
        Unit for voxel size. Default is "micrometer".

    tile_names : Optional[List[str]], optional
        Names for each tile. If None, uses "tile_0", "tile_1", etc.

    channel_names : Optional[List[str]], optional
        Names for each channel. If None, uses "channel_0".

    interest_points_n5 : str or Path, optional
        Path to an existing interestpoints.n5 directory. If provided, its group
        structure is parsed and written into <ViewInterestPoints> in the XML.

    Returns
    -------
    Path
        Path to the created dataset folder containing dataset.xml and dataset.zarr/

    Examples
    --------
    >>> create_bigstitcher_dataset_symlinked(
    ...     zarr_paths=["/data/tile0.zarr", "/data/tile1.zarr"],
    ...     voxel_size=(0.259, 0.259, 1.0),
    ...     output_folder="./my_dataset",
    ...     interest_points_n5="./my_dataset/interestpoints.n5",
    ... )
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    dataset_zarr_path = output_folder / "dataset.zarr"
    dataset_xml_path = output_folder / "dataset.xml"
    dataset_zarr_path.mkdir(exist_ok=True)

    if tile_names is None:
        tile_names = [f"tile_{i}" for i in range(len(zarr_paths))]
    if channel_names is None:
        channel_names = ["channel_0"]

    view_setups = []
    zgroups = []

    for tile_idx, src_path in enumerate(zarr_paths):
        src_path = Path(src_path).resolve()
        tile_name = tile_names[tile_idx]
        tp = 0
        setup_id = tile_idx
        group_name = f"s{setup_id}-t{tp}.zarr"

        # Symlink source zarr into dataset.zarr/
        link_path = dataset_zarr_path / group_name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(src_path)
        print(f"Symlinked: {group_name} -> {src_path}")

        # Read shape from source metadata (no data loaded)
        src_zarr = zarr.open_group(str(src_path), mode='r')
        shape = _read_base_shape(src_zarr)
        z, y, x = shape[-3], shape[-2], shape[-1]

        view_setups.append({
            'id': setup_id,
            'name': f"s{setup_id}-t{tp}",
            'size': (x, y, z),
            'voxel_size': voxel_size,
            'tile_id': tile_idx,
            'tile_name': tile_name,
            'channel_id': 0,
            'timepoint': tp,
        })
        zgroups.append({
            'setup': setup_id,
            'tp': tp,
            'path': group_name,
            'indices': "0 0",
        })

    ip_entries = []
    if interest_points_n5 is not None:
        ip_entries = _parse_interest_points_n5(interest_points_n5)
        print(f"Found {len(ip_entries)} interest point entries in {interest_points_n5}")

    _write_dataset_xml(
        xml_path=dataset_xml_path,
        view_setups=view_setups,
        zgroups=zgroups,
        voxel_unit=voxel_unit,
        channel_names=channel_names,
        interest_points=ip_entries,
    )

    print(f"\nBigStitcher symlinked dataset created at: {output_folder}")
    print(f"  - dataset.xml: {dataset_xml_path}")
    print(f"  - dataset.zarr: {dataset_zarr_path} (symlinks only, no data copied)")

    return output_folder


def _read_base_shape(zarr_group: zarr.Group) -> Tuple[int, ...]:
    """
    Read the base resolution shape from a zarr group.

    Tries multiscales metadata first (path "0"), then falls back to iterating
    subgroups to find the largest array.
    """
    # Try OME-Zarr multiscales metadata
    multiscales = zarr_group.attrs.get("multiscales")
    if multiscales:
        base_path = multiscales[0]["datasets"][0]["path"]
        return zarr_group[base_path].shape

    # Fallback: find "0" or first numeric subgroup
    for key in ["0", "s0"]:
        if key in zarr_group:
            return zarr_group[key].shape

    # Last resort: use the group itself if it's an array
    if hasattr(zarr_group, 'shape'):
        return zarr_group.shape

    raise ValueError(
        f"Cannot determine shape from zarr group. "
        f"Expected OME-Zarr multiscales metadata or a '0'/'s0' resolution level."
    )


def _load_array_lazy(
    arr: Union[zarr.Array, np.ndarray, da.Array, str, Path],
    chunk_size: Tuple[int, int, int]
) -> Tuple[da.Array, Tuple[int, ...], np.dtype]:
    """Load array lazily as a dask array."""
    if isinstance(arr, (str, Path)):
        # Open zarr array lazily
        zarr_arr = zarr.open(str(arr), mode='r')
        dask_arr = da.from_zarr(zarr_arr)
        return dask_arr, zarr_arr.shape, zarr_arr.dtype

    if isinstance(arr, da.Array):
        return arr, arr.shape, arr.dtype

    if isinstance(arr, zarr.Array):
        dask_arr = da.from_zarr(arr)
        return dask_arr, arr.shape, arr.dtype

    if isinstance(arr, np.ndarray):
        # Convert numpy to dask with appropriate chunking
        ndim = arr.ndim
        if ndim == 3:
            chunks = chunk_size
        elif ndim == 4:
            chunks = (1,) + chunk_size
        elif ndim == 5:
            chunks = (1, 1) + chunk_size
        else:
            chunks = "auto"
        dask_arr = da.from_array(arr, chunks=chunks)
        return dask_arr, arr.shape, arr.dtype

    raise TypeError(f"Unsupported array type: {type(arr)}")


def _normalize_shape(shape: Tuple[int, ...]) -> Tuple[Tuple[int, int, int, int, int], int]:
    """Normalize array shape to 5D (t, c, z, y, x)."""
    ndim = len(shape)
    if ndim == 3:
        # (z, y, x) -> (1, 1, z, y, x)
        return (1, 1, shape[0], shape[1], shape[2]), ndim
    elif ndim == 4:
        # (c, z, y, x) -> (1, c, z, y, x)
        return (1, shape[0], shape[1], shape[2], shape[3]), ndim
    elif ndim == 5:
        return shape, ndim
    else:
        raise ValueError(f"Array must be 3D, 4D, or 5D. Got {ndim}D.")


def _normalize_dask_array(
    arr: da.Array,
    original_ndim: int,
    tp: int,
    chunk_size: Tuple[int, int, int]
) -> da.Array:
    """Normalize dask array to 5D (t, c, z, y, x) and extract timepoint."""
    if original_ndim == 3:
        # (z, y, x) -> (1, 1, z, y, x)
        return arr[np.newaxis, np.newaxis, :, :, :]
    elif original_ndim == 4:
        # (c, z, y, x) -> (1, c, z, y, x)
        return arr[np.newaxis, :, :, :, :]
    elif original_ndim == 5:
        # (t, c, z, y, x) -> extract timepoint
        return arr[tp:tp+1, :, :, :, :]
    return arr


def _downsample_dask_array(
    arr: da.Array,
    factor: Tuple[int, int, int],
) -> da.Array:
    """
    Downsample a 3-D or 5-D dask array by ``(fz, fy, fx)`` using block mean.

    Preserves the original dtype.
    """
    fz, fy, fx = factor
    original_dtype = arr.dtype

    if arr.ndim == 3:
        axes = {0: fz, 1: fy, 2: fx}
    else:  # 5D
        axes = {2: fz, 3: fy, 4: fx}

    coarsened = da.coarsen(np.mean, arr, axes, trim_excess=True)
    return coarsened.astype(original_dtype)


def _get_compressor(compression: str, compression_level: int):
    """Get the appropriate compressor."""
    if compression == "zstd":
        try:
            from numcodecs import Zstd
            return Zstd(level=compression_level)
        except ImportError:
            from numcodecs import GZip
            return GZip(level=compression_level)
    elif compression == "gzip":
        from numcodecs import GZip
        return GZip(level=compression_level)
    elif compression == "lz4":
        from numcodecs import LZ4
        return LZ4()
    return None


def _write_resolution_level_3d(
    view_group: zarr.Group,
    level: int,
    dask_data: da.Array,
    chunk_size: Tuple[int, int, int],
    compressor,
    downsampling_factor: Tuple[int, int, int],
) -> zarr.Array:
    """
    Write one 3-D resolution level ``(z, y, x)`` into *view_group*.

    Returns the written ``zarr.Array`` so callers can reload it to break the
    dask computation chain between pyramid levels.
    """
    z, y, x = dask_data.shape
    chunks = (
        min(chunk_size[0], z),
        min(chunk_size[1], y),
        min(chunk_size[2], x),
    )

    level_path = os.path.join(view_group.store.path, view_group.path, str(level))
    level_arr = zarr.open_array(
        level_path,
        mode="w",
        shape=(z, y, x),
        chunks=chunks,
        dtype=dask_data.dtype,
        compressor=compressor,
        dimension_separator="/",
        fill_value=0,
        filters=[],
        order="C",
    )

    da.to_zarr(dask_data.rechunk(chunks), level_arr, overwrite=True, compute=True)

    if level > 0:
        fz, fy, fx = downsampling_factor
        level_arr.attrs["downsamplingFactors"] = [fz, fy, fx]

    return level_arr


def _write_resolution_level_dask(
    view_group: zarr.Group,
    level: int,
    dask_data: da.Array,
    chunk_size: Tuple[int, int, int],
    compressor,
    downsampling_factor: Tuple[int, int, int]
) -> zarr.Array:
    """Write a single resolution level to zarr using dask for parallel copying.

    Returns the written zarr.Array so callers can reload it as a new lazy
    dask array, breaking the computation-graph chain between pyramid levels.
    """
    t, c, z, y, x = dask_data.shape

    # Adjust chunk size to not exceed array dimensions
    chunks = (
        1,  # time
        1,  # channel
        min(chunk_size[0], z),
        min(chunk_size[1], y),
        min(chunk_size[2], x)
    )

    # Get the path to the level within the view group
    level_path = os.path.join(view_group.store.path, view_group.path, str(level))

    # Create the zarr array first with explicit settings matching BigStitcher format
    level_arr = zarr.open_array(
        level_path,
        mode='w',
        shape=(t, c, z, y, x),
        chunks=chunks,
        dtype=dask_data.dtype,
        compressor=compressor,
        dimension_separator='/',
        fill_value=0,
        filters=[],
        order='C'
    )

    # Rechunk dask array to match output chunks for efficient writing
    rechunked = dask_data.rechunk(chunks)

    # Use dask to_zarr to write to the existing zarr array
    da.to_zarr(rechunked, level_arr, overwrite=True, compute=True)

    # Write downsampling factors attribute for non-base levels
    if level > 0:
        fz, fy, fx = downsampling_factor
        level_arr.attrs['downsamplingFactors'] = [fz, fy, fx, 1, 1]

    return level_arr


def _write_multiscale_metadata(
    view_group: zarr.Group,
    base_shape: Tuple[int, int, int],
    downsampling_factors: List[Tuple[int, int, int]],
    voxel_size: Tuple[float, float, float],
    voxel_unit: str,
    ndim: int = 5,
) -> None:
    """
    Write OME-Zarr ``multiscales`` metadata to a view group.

    Parameters
    ----------
    ndim : {3, 5}
        3 for native 3-D zarr groups, 5 for 5-D (t, c, z, y, x) groups.
    """
    z, y, x = base_shape

    if ndim == 3:
        axes = [
            {"type": "space", "name": "z", "unit": voxel_unit, "discrete": False},
            {"type": "space", "name": "y", "unit": voxel_unit, "discrete": False},
            {"type": "space", "name": "x", "unit": voxel_unit, "discrete": False},
        ]
        base_scale = [1.0, 1.0, 1.0]
        base_translation = [0.0, 0.0, 0.0]
        global_scale = [1.0, 1.0, 1.0]

        datasets = [
            {
                "path": "0",
                "coordinateTransformations": [
                    {"scale": base_scale, "type": "scale"},
                    {"translation": base_translation, "type": "translation"},
                ],
            }
        ]
        for i, (fz, fy, fx) in enumerate(downsampling_factors, start=1):
            datasets.append({
                "path": str(i),
                "coordinateTransformations": [
                    {"scale": [float(fz), float(fy), float(fx)], "type": "scale"},
                    {"translation": [(fz - 1) / 2.0, (fy - 1) / 2.0, (fx - 1) / 2.0],
                     "type": "translation"},
                ],
            })

        multiscales = [{
            "name": "/",
            "version": "0.4",
            "axes": axes,
            "datasets": datasets,
            "coordinateTransformations": [{"scale": global_scale, "type": "scale"}],
            "basePath": "",
            "paths": [str(i) for i in range(len(downsampling_factors) + 1)],
            "units": [voxel_unit] * 3,
        }]

    else:  # ndim == 5
        axes = [
            {"type": "time", "name": "t", "unit": "millisecond", "discrete": False},
            {"type": "channel", "name": "c", "discrete": False},
            {"type": "space", "name": "z", "unit": voxel_unit, "discrete": False},
            {"type": "space", "name": "y", "unit": voxel_unit, "discrete": False},
            {"type": "space", "name": "x", "unit": voxel_unit, "discrete": False},
        ]

        datasets = [
            {
                "path": "0",
                "coordinateTransformations": [
                    {"scale": [1.0, 1.0, 1.0, 1.0, 1.0], "type": "scale"},
                    {"translation": [0.0, 0.0, 0.0, 0.0, 0.0], "type": "translation"},
                ],
            }
        ]
        for i, (fz, fy, fx) in enumerate(downsampling_factors, start=1):
            datasets.append({
                "path": str(i),
                "coordinateTransformations": [
                    {"scale": [1.0, 1.0, float(fz), float(fy), float(fx)],
                     "type": "scale"},
                    {"translation": [0.0, 0.0, (fz - 1) / 2.0, (fy - 1) / 2.0,
                                     (fx - 1) / 2.0],
                     "type": "translation"},
                ],
            })

        multiscales = [{
            "name": "/",
            "version": "0.4",
            "axes": axes,
            "datasets": datasets,
            "coordinateTransformations": [
                {"scale": [1.0, 1.0, 1.0, 1.0, 1.0], "type": "scale"}
            ],
            "basePath": "",
            "paths": [str(i) for i in range(len(downsampling_factors) + 1)],
            "units": [voxel_unit] * 5,
        }]

    view_group.attrs["multiscales"] = multiscales


# ── Private helpers: full-tile writers ───────────────────────────────────────


def _write_tile_3d(
    view_group: zarr.Group,
    dask_data: da.Array,
    downsampling_factors: List[Tuple[int, int, int]],
    chunk_size: Tuple[int, int, int],
    compressor,
    voxel_size: Tuple[float, float, float],
    voxel_unit: str,
) -> None:
    """Write a full 3-D multi-resolution pyramid for one tile."""
    z, y, x = dask_data.shape

    print("  Writing level 0 (full resolution, 3D)…")
    _write_resolution_level_3d(
        view_group, 0, dask_data, chunk_size, compressor, (1, 1, 1)
    )

    current_data = dask_data
    prev_factor = (1, 1, 1)
    for level_idx, ds_factor in enumerate(downsampling_factors, start=1):
        rel_factor = tuple(ds_factor[i] // prev_factor[i] for i in range(3))
        print(f"  Writing level {level_idx} (downsample {rel_factor} from level"
              f" {level_idx - 1})…")
        downsampled = _downsample_dask_array(current_data, rel_factor)
        prev_factor = ds_factor
        written_arr = _write_resolution_level_3d(
            view_group, level_idx, downsampled, chunk_size, compressor, ds_factor
        )
        current_data = da.from_zarr(written_arr)

    _write_multiscale_metadata(
        view_group, (z, y, x), downsampling_factors, voxel_size, voxel_unit,
        ndim=3,
    )


def _write_tile_5d(
    view_group: zarr.Group,
    dask_data: da.Array,
    downsampling_factors: List[Tuple[int, int, int]],
    chunk_size: Tuple[int, int, int],
    compressor,
    voxel_size: Tuple[float, float, float],
    voxel_unit: str,
) -> None:
    """Write a full 5-D ``(t, c, z, y, x)`` multi-resolution pyramid for one tile."""
    t, c, z, y, x = dask_data.shape

    print("  Writing level 0 (full resolution, 5D)…")
    _write_resolution_level_dask(
        view_group, 0, dask_data, chunk_size, compressor, (1, 1, 1)
    )

    current_data = dask_data
    prev_factor = (1, 1, 1)
    for level_idx, ds_factor in enumerate(downsampling_factors, start=1):
        rel_factor = tuple(ds_factor[i] // prev_factor[i] for i in range(3))
        print(f"  Writing level {level_idx} (downsample {rel_factor} from level"
              f" {level_idx - 1})…")
        downsampled = _downsample_dask_array(current_data, rel_factor)
        prev_factor = ds_factor
        written_arr = _write_resolution_level_dask(
            view_group, level_idx, downsampled, chunk_size, compressor, ds_factor
        )
        current_data = da.from_zarr(written_arr)

    _write_multiscale_metadata(
        view_group, (z, y, x), downsampling_factors, voxel_size, voxel_unit,
        ndim=5,
    )


def add_interest_points_to_xml(
    xml_path: Union[str, Path],
    interest_points_n5: Union[str, Path],
) -> None:
    """
    Append interest points from an existing interestpoints.n5 into an existing dataset.xml.

    Replaces the contents of the <ViewInterestPoints> element in-place.
    The XML file is updated directly; all other sections are left untouched.

    Parameters
    ----------
    xml_path : str or Path
        Path to the existing BigStitcher dataset.xml to update.

    interest_points_n5 : str or Path
        Path to the interestpoints.n5 directory to parse.

    Examples
    --------
    >>> add_interest_points_to_xml(
    ...     xml_path="./my_dataset/dataset.xml",
    ...     interest_points_n5="./my_dataset/interestpoints.n5",
    ... )
    """
    import xml.etree.ElementTree as ET

    xml_path = Path(xml_path)

    entries = _parse_interest_points_n5(interest_points_n5)
    print(f"Found {len(entries)} interest point entries in {interest_points_n5}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    vip = root.find('ViewInterestPoints')
    if vip is None:
        raise ValueError(f"<ViewInterestPoints> element not found in {xml_path}")

    # Clear existing entries and repopulate
    vip.clear()
    for entry in entries:
        el = ET.SubElement(vip, 'ViewInterestPointsFile')
        el.set('timepoint', str(entry['timepoint']))
        el.set('setup', str(entry['setup']))
        el.set('label', entry['label'])
        el.set('params', entry.get('params', 'manual'))
        el.text = entry['path']

    # Write back with pretty printing
    xml_str = minidom.parseString(
        ET.tostring(root, encoding='unicode')
    ).toprettyxml(indent='  ')
    lines = [line for line in xml_str.split('\n') if line.strip()]
    xml_str = '\n'.join(lines)

    with open(xml_path, 'w', encoding='UTF-8') as f:
        f.write(xml_str)

    print(f"Updated {xml_path} with {len(entries)} interest point entries.")


def _parse_interest_points_n5(n5_path: Union[str, Path]) -> List[dict]:
    """
    Parse an interestpoints.n5 directory and return a list of interest point entries.

    Each entry is a dict with keys: timepoint (int), setup (int), label (str), path (str).
    The path is the N5-relative group path, e.g. "tpId_0_viewSetupId_0/beads".

    Top-level groups are expected to be named "tpId_{tp}_viewSetupId_{setup}".
    Each contains sub-groups named by label (e.g. "beads"), which in turn contain
    "interestpoints" and "correspondences".
    """
    import re
    import warnings

    n5_path = Path(n5_path)
    if not n5_path.exists():
        print(f"Warning: interestpoints.n5 not found at {n5_path}, skipping.")
        return []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        store = zarr.N5Store(str(n5_path))

    root = zarr.open_group(store=store, mode='r')

    pattern = re.compile(r"^tpId_(\d+)_viewSetupId_(\d+)$")
    entries = []

    for group_name in sorted(root.keys()):
        m = pattern.match(group_name)
        if not m:
            continue
        tp = int(m.group(1))
        setup = int(m.group(2))
        view_group = root[group_name]
        for label in sorted(view_group.keys()):
            entries.append({
                "timepoint": tp,
                "setup": setup,
                "label": label,
                "path": f"{group_name}/{label}",
            })

    return entries


def _write_dataset_xml(
    xml_path: Path,
    view_setups: List[dict],
    zgroups: List[dict],
    voxel_unit: str,
    voxel_size: Tuple[float, float, float],
    channel_names: List[str],
    interest_points: Optional[List[dict]] = None,
) -> None:
    """
    Generate and write the BigStitcher ``dataset.xml`` file.

    Parameters
    ----------
    view_setups : list of dict
        Unique per (tile, channel).  Keys: ``id``, ``name``, ``size`` (x,y,z),
        ``voxel_size``, ``tile_id``, ``tile_name``, ``channel_id``.
    zgroups : list of dict
        One entry per (setup, timepoint).  Keys: ``setup``, ``tp``, ``path``,
        ``indices``.
    voxel_size : (vx, vy, vz)
        Used to build the calibration affine in ViewRegistrations.
    channel_names : list of str
        Channel display names indexed by ``channel_id``.
    interest_points : list of dict, optional
        Entries for ``<ViewInterestPoints>``.
    """
    vx, vy, vz = voxel_size

    # ── Root ─────────────────────────────────────────────────────────────────
    spim_data = Element("SpimData")
    spim_data.set("version", "0.2")

    base_path = SubElement(spim_data, "BasePath")
    base_path.set("type", "relative")
    base_path.text = "."

    # ── SequenceDescription ───────────────────────────────────────────────────
    seq_desc = SubElement(spim_data, "SequenceDescription")

    # ImageLoader
    img_loader = SubElement(seq_desc, "ImageLoader")
    img_loader.set("format", "bdv.multimg.zarr")
    img_loader.set("version", "3.0")

    zarr_elem = SubElement(img_loader, "zarr")
    zarr_elem.set("type", "relative")
    zarr_elem.text = "dataset.zarr"

    zgroups_elem = SubElement(img_loader, "zgroups")
    for zg in zgroups:
        zg_elem = SubElement(zgroups_elem, "zgroup")
        zg_elem.set("setup", str(zg["setup"]))
        zg_elem.set("tp", str(zg["tp"]))
        zg_elem.set("path", zg["path"])
        zg_elem.set("indicies", zg["indices"])  # BigStitcher typo preserved

    # ViewSetups
    view_setups_elem = SubElement(seq_desc, "ViewSetups")

    unique_tiles: dict = {}
    unique_channels: set = set()

    for vs in view_setups:
        vs_elem = SubElement(view_setups_elem, "ViewSetup")

        SubElement(vs_elem, "id").text = str(vs["id"])
        SubElement(vs_elem, "name").text = vs["name"]

        size_elem = SubElement(vs_elem, "size")
        size_elem.text = f"{vs['size'][0]} {vs['size'][1]} {vs['size'][2]}"

        voxel_elem = SubElement(vs_elem, "voxelSize")
        SubElement(voxel_elem, "unit").text = voxel_unit
        vsize_elem = SubElement(voxel_elem, "size")
        vsize_elem.text = (
            f"{vs['voxel_size'][0]} {vs['voxel_size'][1]} {vs['voxel_size'][2]}"
        )

        attrs_elem = SubElement(vs_elem, "attributes")
        SubElement(attrs_elem, "illumination").text = "0"
        SubElement(attrs_elem, "channel").text = str(vs["channel_id"])
        SubElement(attrs_elem, "tile").text = str(vs["tile_id"])
        SubElement(attrs_elem, "angle").text = "0"

        unique_tiles[vs["tile_id"]] = vs["tile_name"]
        unique_channels.add(vs["channel_id"])

    # Illumination attributes (single illumination)
    illum_attrs = SubElement(view_setups_elem, "Attributes")
    illum_attrs.set("name", "illumination")
    illum = SubElement(illum_attrs, "Illumination")
    SubElement(illum, "id").text = "0"
    SubElement(illum, "name").text = "0"

    # Channel attributes
    ch_attrs = SubElement(view_setups_elem, "Attributes")
    ch_attrs.set("name", "channel")
    for ch_id in sorted(unique_channels):
        ch = SubElement(ch_attrs, "Channel")
        SubElement(ch, "id").text = str(ch_id)
        SubElement(ch, "name").text = (
            channel_names[ch_id] if ch_id < len(channel_names) else str(ch_id)
        )

    # Tile attributes
    tile_attrs = SubElement(view_setups_elem, "Attributes")
    tile_attrs.set("name", "tile")
    for tile_id in sorted(unique_tiles):
        tile = SubElement(tile_attrs, "Tile")
        SubElement(tile, "id").text = str(tile_id)
        SubElement(tile, "name").text = unique_tiles[tile_id]

    # Angle attributes (single angle)
    angle_attrs = SubElement(view_setups_elem, "Attributes")
    angle_attrs.set("name", "angle")
    angle = SubElement(angle_attrs, "Angle")
    SubElement(angle, "id").text = "0"
    SubElement(angle, "name").text = "0"

    # Timepoints — derived from zgroups
    unique_tps = sorted({zg["tp"] for zg in zgroups})
    timepoints = SubElement(seq_desc, "Timepoints")
    timepoints.set("type", "pattern")
    SubElement(timepoints, "integerpattern").text = ",".join(str(tp) for tp in unique_tps)

    SubElement(seq_desc, "MissingViews")

    # ── ViewRegistrations ─────────────────────────────────────────────────────
    # One registration per unique (tp, setup) pair, ordered by tp then setup.
    calibration_affine = (
        f"{vx} 0.0 0.0 0.0 0.0 {vy} 0.0 0.0 0.0 0.0 {vz} 0.0"
    )

    view_regs = SubElement(spim_data, "ViewRegistrations")
    seen_regs: set = set()
    for zg in sorted(zgroups, key=lambda z: (z["tp"], z["setup"])):
        key = (zg["tp"], zg["setup"])
        if key in seen_regs:
            continue
        seen_regs.add(key)

        vr = SubElement(view_regs, "ViewRegistration")
        vr.set("timepoint", str(zg["tp"]))
        vr.set("setup", str(zg["setup"]))

        vt = SubElement(vr, "ViewTransform")
        vt.set("type", "affine")
        SubElement(vt, "Name").text = "calibration"
        SubElement(vt, "affine").text = calibration_affine

    # ── Optional sections ─────────────────────────────────────────────────────
    vip_elem = SubElement(spim_data, "ViewInterestPoints")
    for entry in (interest_points or []):
        vip_file = SubElement(vip_elem, "ViewInterestPointsFile")
        vip_file.set("timepoint", str(entry["timepoint"]))
        vip_file.set("setup", str(entry["setup"]))
        vip_file.set("label", entry["label"])
        vip_file.set("params", entry.get("params", "manual"))
        vip_file.text = entry["path"]

    SubElement(spim_data, "BoundingBoxes")
    SubElement(spim_data, "PointSpreadFunctions")
    SubElement(spim_data, "StitchingResults")
    SubElement(spim_data, "IntensityAdjustments")

    # ── Serialise ─────────────────────────────────────────────────────────────
    xml_str = minidom.parseString(
        tostring(spim_data, encoding="unicode")
    ).toprettyxml(indent="  ")

    lines = [line for line in xml_str.split("\n") if line.strip()]
    xml_str = "\n".join(lines)

    with open(xml_path, "w", encoding="UTF-8") as f:
        f.write(xml_str)
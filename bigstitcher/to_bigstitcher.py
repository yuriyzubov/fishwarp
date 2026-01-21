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

                # Write downsampled levels
                current_data = dask_data
                for level_idx, ds_factor in enumerate(downsampling_factors, start=1):
                    print(f"  Writing level {level_idx} (downsample {ds_factor})...")
                    downsampled = _downsample_dask_array(current_data, ds_factor)
                    _write_resolution_level_dask(
                        view_group=view_group,
                        level=level_idx,
                        dask_data=downsampled,
                        chunk_size=chunk_size,
                        compressor=compressor,
                        downsampling_factor=ds_factor
                    )

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

        # Generate the XML file
        _write_dataset_xml(
            xml_path=dataset_xml_path,
            view_setups=view_setups,
            zgroups=zgroups,
            voxel_unit=voxel_unit,
            channel_names=channel_names
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


def _downsample_dask_array(arr: da.Array, factor: Tuple[int, int, int]) -> da.Array:
    """Downsample a 5D dask array by the given factors (z, y, x), preserving dtype."""
    fz, fy, fx = factor
    original_dtype = arr.dtype

    # Use dask coarsen for efficient downsampling with mean aggregation
    coarsened = da.coarsen(
        np.mean,
        arr,
        {2: fz, 3: fy, 4: fx},  # Axes: t=0, c=1, z=2, y=3, x=4
        trim_excess=True
    )

    # Cast back to original dtype to preserve data type consistency
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


def _write_resolution_level_dask(
    view_group: zarr.Group,
    level: int,
    dask_data: da.Array,
    chunk_size: Tuple[int, int, int],
    compressor,
    downsampling_factor: Tuple[int, int, int]
) -> None:
    """Write a single resolution level to zarr using dask for parallel copying."""
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


def _write_multiscale_metadata(
    view_group: zarr.Group,
    base_shape: Tuple[int, int, int],
    downsampling_factors: List[Tuple[int, int, int]],
    voxel_size: Tuple[float, float, float],
    voxel_unit: str
) -> None:
    """Write OME-Zarr multiscale metadata to the view group."""
    vx, vy, vz = voxel_size
    z, y, x = base_shape

    # Build datasets list
    datasets = [
        {
            "path": "0",
            "coordinateTransformations": [
                {"scale": [1.0, 1.0, 1.0, 1.0, 1.0], "type": "scale"},
                {"translation": [0.0, 0.0, 0.0, 0.0, 0.0], "type": "translation"}
            ]
        }
    ]

    for i, (fz, fy, fx) in enumerate(downsampling_factors, start=1):
        datasets.append({
            "path": str(i),
            "coordinateTransformations": [
                {"scale": [1.0, 1.0, float(fz), float(fy), float(fx)], "type": "scale"},
                {"translation": [0.0, 0.0, (fz-1)/2.0, (fy-1)/2.0, (fx-1)/2.0], "type": "translation"}
            ]
        })

    multiscales = [{
        "name": "/",
        "version": "0.4",
        "axes": [
            {"type": "time", "name": "t", "unit": "millisecond", "discrete": False},
            {"type": "channel", "name": "c", "discrete": False},
            {"type": "space", "name": "z", "unit": voxel_unit, "discrete": False},
            {"type": "space", "name": "y", "unit": voxel_unit, "discrete": False},
            {"type": "space", "name": "x", "unit": voxel_unit, "discrete": False}
        ],
        "datasets": datasets,
        "coordinateTransformations": [
            {"scale": [1.0, 1.0, 1.0, 1.0, 1.0], "type": "scale"}
        ],
        "basePath": "",
        "paths": [str(i) for i in range(len(downsampling_factors) + 1)],
        "units": [voxel_unit] * 5
    }]

    view_group.attrs['multiscales'] = multiscales


def _write_dataset_xml(
    xml_path: Path,
    view_setups: List[dict],
    zgroups: List[dict],
    voxel_unit: str,
    channel_names: List[str]
) -> None:
    """Generate the BigStitcher XML file."""
    # Root element
    spim_data = Element('SpimData')
    spim_data.set('version', '0.2')

    # BasePath
    base_path = SubElement(spim_data, 'BasePath')
    base_path.set('type', 'relative')
    base_path.text = '.'

    # SequenceDescription
    seq_desc = SubElement(spim_data, 'SequenceDescription')

    # ImageLoader
    img_loader = SubElement(seq_desc, 'ImageLoader')
    img_loader.set('format', 'bdv.multimg.zarr')
    img_loader.set('version', '3.0')

    zarr_elem = SubElement(img_loader, 'zarr')
    zarr_elem.set('type', 'relative')
    zarr_elem.text = 'dataset.zarr'

    zgroups_elem = SubElement(img_loader, 'zgroups')
    for zg in zgroups:
        zgroup_elem = SubElement(zgroups_elem, 'zgroup')
        zgroup_elem.set('setup', str(zg['setup']))
        zgroup_elem.set('tp', str(zg['tp']))
        zgroup_elem.set('path', zg['path'])
        zgroup_elem.set('indicies', zg['indices'])

    # ViewSetups
    view_setups_elem = SubElement(seq_desc, 'ViewSetups')

    # Collect unique tiles and channels
    unique_tiles = {}
    unique_channels = set()

    for vs in view_setups:
        # ViewSetup element
        vs_elem = SubElement(view_setups_elem, 'ViewSetup')

        id_elem = SubElement(vs_elem, 'id')
        id_elem.text = str(vs['id'])

        name_elem = SubElement(vs_elem, 'name')
        name_elem.text = vs['name']

        size_elem = SubElement(vs_elem, 'size')
        size_elem.text = f"{vs['size'][0]} {vs['size'][1]} {vs['size'][2]}"

        voxel_elem = SubElement(vs_elem, 'voxelSize')
        unit_elem = SubElement(voxel_elem, 'unit')
        unit_elem.text = voxel_unit
        vsize_elem = SubElement(voxel_elem, 'size')
        vsize_elem.text = f"{vs['voxel_size'][0]} {vs['voxel_size'][1]} {vs['voxel_size'][2]}"

        attrs_elem = SubElement(vs_elem, 'attributes')

        illum_elem = SubElement(attrs_elem, 'illumination')
        illum_elem.text = '0'

        channel_elem = SubElement(attrs_elem, 'channel')
        channel_elem.text = str(vs['channel_id'])

        tile_elem = SubElement(attrs_elem, 'tile')
        tile_elem.text = str(vs['tile_id'])

        angle_elem = SubElement(attrs_elem, 'angle')
        angle_elem.text = '0'

        unique_tiles[vs['tile_id']] = vs['tile_name']
        unique_channels.add(vs['channel_id'])

    # Illumination attributes
    illum_attrs = SubElement(view_setups_elem, 'Attributes')
    illum_attrs.set('name', 'illumination')
    illum = SubElement(illum_attrs, 'Illumination')
    illum_id = SubElement(illum, 'id')
    illum_id.text = '0'
    illum_name = SubElement(illum, 'name')
    illum_name.text = '0'

    # Channel attributes
    channel_attrs = SubElement(view_setups_elem, 'Attributes')
    channel_attrs.set('name', 'channel')
    for ch_id in sorted(unique_channels):
        ch = SubElement(channel_attrs, 'Channel')
        ch_id_elem = SubElement(ch, 'id')
        ch_id_elem.text = str(ch_id)
        ch_name_elem = SubElement(ch, 'name')
        ch_name_elem.text = channel_names[ch_id] if ch_id < len(channel_names) else str(ch_id)

    # Tile attributes
    tile_attrs = SubElement(view_setups_elem, 'Attributes')
    tile_attrs.set('name', 'tile')
    for tile_id in sorted(unique_tiles.keys()):
        tile = SubElement(tile_attrs, 'Tile')
        tile_id_elem = SubElement(tile, 'id')
        tile_id_elem.text = str(tile_id)
        tile_name_elem = SubElement(tile, 'name')
        tile_name_elem.text = unique_tiles[tile_id]

    # Angle attributes
    angle_attrs = SubElement(view_setups_elem, 'Attributes')
    angle_attrs.set('name', 'angle')
    angle = SubElement(angle_attrs, 'Angle')
    angle_id = SubElement(angle, 'id')
    angle_id.text = '0'
    angle_name = SubElement(angle, 'name')
    angle_name.text = '0'

    # Timepoints
    timepoints = SubElement(seq_desc, 'Timepoints')
    timepoints.set('type', 'pattern')
    tp_pattern = SubElement(timepoints, 'integerpattern')
    unique_tps = sorted(set(vs['timepoint'] for vs in view_setups))
    tp_pattern.text = ', '.join(str(tp) for tp in unique_tps)

    # MissingViews (empty)
    SubElement(seq_desc, 'MissingViews')

    # ViewRegistrations (identity transforms)
    view_regs = SubElement(spim_data, 'ViewRegistrations')
    for vs in view_setups:
        vr = SubElement(view_regs, 'ViewRegistration')
        vr.set('timepoint', str(vs['timepoint']))
        vr.set('setup', str(vs['id']))

        vt = SubElement(vr, 'ViewTransform')
        vt.set('type', 'affine')

        name = SubElement(vt, 'Name')
        name.text = 'calibration'

        affine = SubElement(vt, 'affine')
        affine.text = '1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0'

    # Empty sections
    SubElement(spim_data, 'ViewInterestPoints')
    SubElement(spim_data, 'BoundingBoxes')
    SubElement(spim_data, 'PointSpreadFunctions')
    SubElement(spim_data, 'StitchingResults')
    SubElement(spim_data, 'IntensityAdjustments')

    # Write with pretty printing
    xml_str = minidom.parseString(
        tostring(spim_data, encoding='unicode')
    ).toprettyxml(indent='  ')

    # Remove extra blank lines
    lines = [line for line in xml_str.split('\n') if line.strip()]
    xml_str = '\n'.join(lines)

    with open(xml_path, 'w', encoding='UTF-8') as f:
        f.write(xml_str)



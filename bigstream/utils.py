def crop_zarr_sample(zarr_path, crop_center, box_size, output_path=None, dataset_name='s0'):
    """
    # TODO: Replace with a standard method crop ROI methods

    Extract a crop from a zarr dataset at specified voxel coordinates.
    
    Parameters:
    -----------
    zarr_path : str
        Path to the source zarr file
    crop_center : tuple
        Center coordinates (z, y, x) for the crop
    box_size : list or tuple
        Full box size in each dimension [z_size, y_size, x_size]
    output_path : str, optional
        Path to save the cropped zarr. If None, returns the data array
    dataset_name : str, default 's0'
        Name of the dataset within the zarr file
        
    Returns:
    --------
    numpy.ndarray or None
        If output_path is None, returns the cropped data array.
        Otherwise saves to zarr and returns None.
    """
    import zarr
    import os
    
    # Calculate half box size for each dimension
    half_boxsize = [size // 2 for size in box_size]
    
    # Open source zarr
    zg_src = zarr.open(zarr_path, mode='r')
    
    # Construct slice for cropping
    slices = tuple(slice(center - halo, center + halo) 
                  for center, halo in zip(crop_center, half_boxsize))
    
    # Fetch cropped data
    src_sl = zg_src[dataset_name][slices]
    
    # If no output path specified, return the data
    if output_path is None:
        return src_sl
    
    # Otherwise, save to zarr file
    zg_dest = zarr.open(output_path, mode='a')
    
    # Write into zarr
    zarr_dest = zg_dest.require_dataset(
        name=dataset_name,
        shape=src_sl.shape,
        dtype=src_sl.dtype,
        chunks=(128, 128, 128),
        dimension_separator='/'
    )
    zarr_dest[:] = src_sl[:]
    
    # Copy and modify attributes,
    # TODO: currently only for s0 level, do for an arbitrary level with translation or replace with a standard method
    z_attrs = dict(zg_src.attrs)
    z_attrs['multiscales'][0]['datasets'] = [zg_src.attrs['multiscales'][0]['datasets'][0]]
    zg_dest.attrs['multiscales'] = z_attrs['multiscales']
    
    return None
import zarr

def create_foreground_mask(
    input_zarr_path: str, 
    output_zarr_path: str, 
    restrict: float = 0.1, 
    min_object_size: int = 1000, 
    closing_radius: int = 2
) -> zarr.Group:
    """
    Create a foreground mask from a 3D zarr array using Otsu thresholding.
    
    Parameters:
    -----------
    input_zarr_path : str
        Path to input zarr array (e.g., '/path/to/input.zarr/s3')
    output_zarr_path : str  
        Path to output zarr array (e.g., '/path/to/output.zarr')
    restrict : float, default=0.1
        Factor to multiply Otsu threshold (lower = more permissive)
    min_object_size : int, default=1000
        Minimum size of objects to keep in voxels (removes small noise)
    closing_radius : int, default=2
        Radius of ball structuring element for morphological closing in voxels
        
    Returns:
    --------
    zarr.Group
        The zarr group containing the mask dataset at 's0'
    """
    from skimage.filters import threshold_otsu
    from scipy import ndimage
    from skimage.morphology import remove_small_objects, binary_closing, ball
    
    # Load input data
    img_src = zarr.open(input_zarr_path, mode='r')[:]
    
    # Create initial threshold mask
    thresh = threshold_otsu(img_src) * restrict
    foreground_mask = img_src > thresh
    
    # Clean up in 3D
    # Remove small 3D noise
    foreground_mask = remove_small_objects(foreground_mask, min_size=min_object_size)
    
    # Fill holes and smooth in 3D
    foreground_mask = binary_closing(foreground_mask, ball(closing_radius))
    foreground_mask = ndimage.binary_fill_holes(foreground_mask)
    
    # Store as zarr array
    output_mask_zarr = zarr.open(output_zarr_path)
    output_mask = output_mask_zarr.require_dataset(
        's0', 
        shape=foreground_mask.shape, 
        chunks=(128, 128, 128), 
        dtype='uint8'
    )
    output_mask[:] = foreground_mask.astype('uint8')[:]
    
    # Return the zarr group, not the dataset itself
    return output_mask_zarr
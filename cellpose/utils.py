import zarr
import os
import json
import argparse
from typing import Dict, Any, Tuple, List

def get_foreground_mask(
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

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return json.load(f)

def run_cellpose_distributed(config_path : str) -> Tuple[zarr.Array, List]:
    """
    Run distributed cellpose segmentation based on config.
    
    Parameters:
    -----------
    config : str
        Configuration dictionary with the following required keys:
        
        src_path : str
            Path to source zarr array (e.g., '/path/to/input.zarr/s0')
        dest_path : str  
            Path to destination directory for output zarr
        foreground_path : str, optional
            Path to foreground mask zarr array (e.g., '/path/to/mask.zarr/s0')
        blocksize : list of int, default [256, 256, 256]
            Block dimensions for distributed processing [z, y, x]
        model_kwargs : dict
            Cellpose model parameters:
            - gpu : bool - Use GPU acceleration
            - pretrained_model : str, optional - Path to custom model
        eval_kwargs : dict
            Cellpose evaluation parameters:
            - diameter : int - Expected cell diameter in pixels
            - z_axis : int - Which axis is Z (0, 1, or 2)  
            - channels : [int, int] - [cytoplasm_channel, nucleus_channel]
            - min_size : int - Minimum object size in pixels
            - anisotropy : float or None - Z-anisotropy factor
            - stitch_threshold : float, optional - Threshold for stitching blocks
            - do_3D : bool - Use 3D segmentation
        cluster_kwargs : dict
            Cluster/compute parameters. For LSF:
            - ncpus : int - CPUs per worker
            - min_workers : int - Minimum number of workers
            - max_workers : int - Maximum number of workers  
            - queue : str - LSF queue name
            - job_extra_directives : [str] - Additional LSF directives
            - log_directory : str - Directory for job logs
            For local execution:
            - n_workers : int - Number of workers
            - memory_limit : str - Memory limit per worker
            - threads_per_worker : int - Threads per worker
            
    Returns:
    --------
    tuple
        (segments, boxes) - zarr array with labels and bounding boxes
    """
    from cellpose.contrib.distributed_segmentation import distributed_eval

    with open(config_path, 'r') as f:
        config = json.load(f)
    # Load source volume
    src_volume = zarr.open(store=config['src_path'], mode='r')
    print(f"Source volume shape: {src_volume.shape}")
    print(f"Source volume type: {type(src_volume)}")
    
    # Load foreground mask if provided
    foreground_mask = None
    if 'foreground_path' in config and config['foreground_path']:
        foreground_mask = zarr.open(store=config['foreground_path'], mode='r')[:]
        print(f"Foreground mask shape: {foreground_mask.shape}")
        print(f"Foreground mask type: {type(foreground_mask)}")
    
    # Extract parameters from config
    model_kwargs = config.get('model_kwargs', {})
    eval_kwargs = config.get('eval_kwargs', {})
    cluster_kwargs = config.get('cluster_kwargs', {})
    blocksize = tuple(config.get('blocksize', [128, 128, 128]))
    
    # Run segmentation
    segments, boxes = distributed_eval(
        input_zarr=src_volume,
        blocksize=blocksize,
        write_path=os.path.join(config['dest_path'], 's0'),
        mask=foreground_mask,
        model_kwargs=model_kwargs,
        eval_kwargs=eval_kwargs,
        cluster_kwargs=cluster_kwargs,
    )
    
    print(f"finished segmenting data in {config['src_path']}")
    
    return segments, boxes

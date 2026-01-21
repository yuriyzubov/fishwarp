import os
from to_bigstitcher import create_bigstitcher_dataset
import zarr 

if __name__ == "__main__":
    # Example usage with dask arrays
    import dask.array as da

    # Create sample dask arrays (lazy - not computed until needed)
    # tiles/ could be numpy array or zarr arrays.
    tile1 = da.random.randint(0, 255, size=(50, 256, 256), dtype=np.uint8, chunks=(64, 128, 128))
    tile2 = da.random.randint(0, 255, size=(50, 256, 256), dtype=np.uint8, chunks=(64, 128, 128))

    dest_path = 'path/to/output/bigstitcher/directory/'
    
    # Convert to BigStitcher format with parallel processing
    output_path = create_bigstitcher_dataset(
        zarr_arrays=[tile1, tile2],
        voxel_size=(0.259, 0.259, 1.0),  # x, y, z in micrometers
        output_folder=dest_path,
        tile_names=["tile_0", "tile_1"],
        voxel_unit="micrometer",
        n_workers=16,
        threads_per_worker=2,
        memory_limit="30GB"
    )

    print(f"\nDataset created at: {output_path}")

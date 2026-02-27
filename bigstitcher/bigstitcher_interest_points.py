"""
BigStitcher Interest Points Writer

This module provides functionality to write interest points from CSV files (or DataFrames)
into a BigStitcher-compatible interestpoints.n5 container.

The N5 structure produced matches BigStitcher's expected format:
    interestpoints.n5/
        tpId_0_viewSetupId_0/
            beads/
                interestpoints/   <- loc [N,3] float64 + id [N,1] uint64
                correspondences/  <- empty, required by BigStitcher
        tpId_0_viewSetupId_1/
            ...
"""

import numpy as np
import pandas as pd
import zarr
import warnings
from pathlib import Path
from typing import Dict, List, Union


def get_bigstitcher_interest_points(
    setups: Union[
        List[Union[pd.DataFrame, str, Path]],
        Dict[int, Union[pd.DataFrame, str, Path]],
    ],
    output_path: Union[str, Path],
    label: str = "beads",
    timepoint: int = 0,
    x_col: str = "COM X (pixels)",
    y_col: str = "COM Y (pixels)",
    z_col: str = "COM Z (pixels)",
    id_col: str = "Object ID",
    chunk_size: int = 300000,
    overwrite: bool = True,
) -> Path:
    """
    Write interest points from DataFrames or CSV files into a BigStitcher N5 container.

    Parameters
    ----------
    setups : List or Dict
        Interest point data for each view setup. Can be:
        - A list of DataFrames or CSV file paths, where index in the list becomes
          the viewSetupId (0, 1, 2, ...).
        - A dict mapping viewSetupId (int) -> DataFrame or CSV file path.

    output_path : str or Path
        Path where interestpoints.n5 will be created (or overwritten).

    label : str, optional
        Interest point label used as the group name inside each view.
        Default is "beads".

    timepoint : int, optional
        Timepoint index. Default is 0.

    x_col : str, optional
        Column name for X coordinates (pixels). Default is "COM X (pixels)".

    y_col : str, optional
        Column name for Y coordinates (pixels). Default is "COM Y (pixels)".

    z_col : str, optional
        Column name for Z coordinates (pixels). Default is "COM Z (pixels)".

    id_col : str, optional
        Column name for point IDs. Default is "Object ID".

    chunk_size : int, optional
        Chunk size along the N (points) axis for zarr arrays. Default is 300000.

    overwrite : bool, optional
        If True, remove and recreate output_path if it already exists. Default is True.

    Returns
    -------
    Path
        Path to the created interestpoints.n5 directory.

    Examples
    --------
    >>> # From CSV files, two setups
    >>> get_bigstitcher_interest_points(
    ...     setups=["center_of_masses_em.csv", "center_of_masses_confocal.csv"],
    ...     output_path="./dataset/interestpoints.n5",
    ... )

    >>> # From DataFrames with a custom label
    >>> import pandas as pd
    >>> df_em = pd.read_csv("center_of_masses_em.csv")
    >>> df_confocal = pd.read_csv("center_of_masses_confocal.csv")
    >>> get_bigstitcher_interest_points(
    ...     setups={0: df_em, 1: df_confocal},
    ...     output_path="./dataset/interestpoints.n5",
    ...     label="nuclei",
    ... )
    """
    output_path = Path(output_path)

    if overwrite and output_path.exists():
        import shutil
        print(f"Removing existing {output_path}")
        shutil.rmtree(output_path)

    # Normalise setups to {setup_id: source}
    if isinstance(setups, list):
        setup_map = {i: src for i, src in enumerate(setups)}
    else:
        setup_map = dict(setups)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        store = zarr.N5Store(str(output_path))

    root = zarr.group(store=store)

    print(f"Creating interestpoints.n5 at {output_path}")

    for setup_id, source in setup_map.items():
        df = _load_dataframe(source)
        print(f"\n{'=' * 60}")
        print(f"Processing viewSetupId={setup_id}  ({len(df)} points)")
        print(f"{'=' * 60}")
        _write_setup(
            df=df,
            setup_id=setup_id,
            timepoint=timepoint,
            label=label,
            root=root,
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
            id_col=id_col,
            chunk_size=chunk_size,
        )

    print(f"\nDone. interestpoints.n5 written to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_dataframe(source: Union[pd.DataFrame, str, Path]) -> pd.DataFrame:
    """Return a DataFrame from a DataFrame, CSV path, or string path."""
    if isinstance(source, pd.DataFrame):
        return source
    return pd.read_csv(source)


def _write_setup(
    df: pd.DataFrame,
    setup_id: int,
    timepoint: int,
    label: str,
    root: zarr.Group,
    x_col: str,
    y_col: str,
    z_col: str,
    id_col: str,
    chunk_size: int,
) -> None:
    """Write interestpoints and correspondences groups for one view setup."""
    x = df[x_col].values
    y = df[y_col].values
    z = df[z_col].values
    n_points = len(df)

    print(f"  X range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"  Y range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"  Z range: [{z.min():.2f}, {z.max():.2f}]")

    # loc array: [N, 3] as float64
    loc_data = np.column_stack([x, y, z]).astype(np.float64)

    # id array: [N, 1] as uint64
    id_data = df[id_col].values.reshape(-1, 1).astype(np.uint64)

    base_path = f"tpId_{timepoint}_viewSetupId_{setup_id}/{label}"

    # --- interestpoints group ---
    ip_group = root.require_group(f"{base_path}/interestpoints")
    ip_group.attrs["pointcloud"] = "1.0.0"
    ip_group.attrs["type"] = "list"
    ip_group.attrs["list version"] = "1.0.0"

    compressor = zarr.GZip(level=-1)

    ip_group.create_dataset(
        "loc",
        data=loc_data,
        chunks=(min(chunk_size, n_points), 3),
        dtype="<f8",
        compressor=compressor,
        overwrite=True,
    )
    ip_group.create_dataset(
        "id",
        data=id_data,
        chunks=(min(chunk_size, n_points), 1),
        dtype="<u8",
        compressor=compressor,
        overwrite=True,
    )

    print(f"  ✓ interestpoints written  (loc {loc_data.shape}, id {id_data.shape})")

    # --- correspondences group (empty, required by BigStitcher) ---
    corr_group = root.require_group(f"{base_path}/correspondences")
    corr_group.attrs["correspondences"] = "1.0.0"
    corr_group.attrs["idMap"] = {}

    print(f"  ✓ correspondences written")

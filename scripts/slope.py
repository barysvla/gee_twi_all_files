from __future__ import annotations

from typing import Any, Literal
import contextlib
import io
import logging
import os
import tempfile

import ee
import geemap
import numpy as np
import rasterio


Mode = Literal["cloud", "local"]


def compute_slope(
    dem: ee.Image,
    *,
    mode: Mode = "cloud",
    grid: dict[str, Any] | None = None,
    band_name: str = "slope",
    quiet: bool = True,
) -> ee.Image | np.ndarray:
    """
    Calculation of terrain slope (in degrees) from a digital elevation model (DEM).

    The function supports two execution modes:

    - "cloud": the slope is computed and returned as an ee.Image within
      the Google Earth Engine environment.

    - "local": the slope is computed in Earth Engine, exported to a
      GeoTIFF aligned to a predefined computational grid, read into
      a NumPy array, and post-processed (NoData handling and masking).

    Parameters
    ----------
    dem : ee.Image
        Input DEM represented as an Earth Engine image.
    mode : {"cloud", "local"}
        Execution mode controlling the output type.
    grid : dict or None
        Grid definition required for local mode. The dictionary must
        define the target spatial reference, export region, and
        NoData mask to ensure spatial consistency with other layers.
    band_name : str
        Name assigned to the output slope band.
    quiet : bool
        If True, suppresses verbose logging during export.

    Returns
    -------
    ee.Image or np.ndarray
        Slope raster in degrees, either as an Earth Engine image
        (cloud mode) or as a NumPy array (local mode).
    """

    # Slope computation using the built-in Earth Engine terrain operator
    slope_img = ee.Terrain.slope(dem).toFloat().rename(band_name)

    # Cloud execution: return Earth Engine image directly
    if mode == "cloud":
        return slope_img

    # Validate execution mode
    if mode != "local":
        raise ValueError(f"Unsupported mode: {mode!r}. Use 'cloud' or 'local'.")

    if grid is None:
        raise ValueError("Grid definition is required in local mode.")

    # Verification of required grid components
    for key in ("projection_info", "region_used", "nodata_mask"):
        if key not in grid:
            raise KeyError(f"Grid definition is missing required key: '{key}'")

    proj_info = grid["projection_info"]
    crs_str = proj_info["crs"]
    crs_transform_list = proj_info.get("transform", None)
    region_aligned = grid["region_used"]

    # Definition of export parameters ensuring spatial alignment
    export_kwargs: dict[str, Any] = {
        "region": region_aligned,
        "file_per_band": False,
        "crs": crs_str,
    }

    if crs_transform_list is not None:
        export_kwargs["crs_transform"] = crs_transform_list
    else:
        if "scale_m" not in grid or grid["scale_m"] is None:
            raise KeyError(
                "Grid must define 'scale_m' when no affine transform is provided."
            )
        export_kwargs["scale"] = float(grid["scale_m"])

    # Optional suppression of verbose logging during export
    previous_levels: dict[str, int] = {}
    if quiet:
        for name in ("google", "googleapiclient", "geemap"):
            logger = logging.getLogger(name)
            previous_levels[name] = logger.level
            logger.setLevel(logging.ERROR)

    def _ee_export(img: ee.Image, filename: str, **kwargs: Any) -> None:
        """
        Export of a single-band Earth Engine image to GeoTIFF.
        """
        if quiet:
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                geemap.ee_export_image(img, filename=filename, **kwargs)
        else:
            geemap.ee_export_image(img, filename=filename, **kwargs)

    try:
        # Temporary directory used for intermediate raster export
        with tempfile.TemporaryDirectory() as tmp_dir:
            slope_tif = os.path.join(tmp_dir, "slope.tif")

            _ee_export(slope_img, filename=slope_tif, **export_kwargs)

            if not os.path.exists(slope_tif):
                raise RuntimeError(
                    f"Earth Engine export failed; file was not created: {slope_tif}"
                )

            # Reading exported GeoTIFF into NumPy
            with rasterio.open(slope_tif) as src:
                slope_np = src.read(1).astype(np.float32)
                nodata_val = src.nodata

    finally:
        # Restore original logging levels
        if quiet:
            for name, lvl in previous_levels.items():
                logging.getLogger(name).setLevel(lvl)

    # Conversion of NoData values to NaN
    if nodata_val is not None:
        slope_np = np.where(np.isclose(slope_np, nodata_val), np.nan, slope_np)

    # Application of DEM NoData mask to ensure spatial consistency
    slope_np = np.where(
        np.asarray(grid["nodata_mask"], dtype=bool),
        np.nan,
        slope_np,
    )

    # Optional consistency check of raster dimensions
    dem_ref = grid.get("dem_elevations", None)
    if dem_ref is not None and slope_np.shape != dem_ref.shape:
        raise ValueError(
            f"Grid mismatch detected: slope {slope_np.shape} vs DEM {dem_ref.shape}"
        )

    return slope_np

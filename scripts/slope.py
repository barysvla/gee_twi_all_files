import ee
import geemap
import numpy as np

import rasterio
import tempfile
import os
import time
import shutil

def compute_slope(dem):
    """
    Computation of terrain slope based on DEM.
    """
    slope = ee.Terrain.slope(dem).rename("Slope")
    
    return slope

def slope_ee_to_numpy_on_grid(grid: dict, ee_dem_grid: ee.Image) -> np.ndarray:
    """
    Convert EE slope (computed from ee_dem_grid) to a NumPy array on the SAME grid as DEM.
    Uses primitive CRS string + transform list from `grid["projection_info"]`.

    Steps:
      1) Read CRS (string) and transform (list) + aligned region from grid.
      2) Build slope image from ee_dem_grid on the same EE grid (float).
      3) Export slope GeoTIFF using the same (crs, crsTransform) as DEM export.
      4) Read TIFF with rasterio -> NumPy, convert nodata to NaN, enforce DEM nodata mask.

    Returns:
      - slope_np: 2D numpy array (float32) aligned with DEM grid (same shape as grid["dem"])
    """
    # 1) Get primitive projection info from grid (STRING CRS + LIST transform)
    proj_info = grid["projection_info"]                # {'crs': 'EPSG:xxxx' or WKT string, 'transform': [a,b,c,d,e,f]}
    crs_str = proj_info["crs"]                         # STRING (do NOT pass rasterio.CRS)
    crs_transform_list = proj_info["transform"]        # LIST
    region_aligned = grid["region_used"]               # ee.Geometry aligned to DEM grid

    # 2) Build slope image from the already aligned ee_dem_grid
    slope_img = ee.Terrain.slope(ee_dem_grid).toFloat().rename("Slope")

    # 3) Export slope GeoTIFF using same (crs, crsTransform)
    tmp_dir = tempfile.mkdtemp()
    slope_tif = os.path.join(tmp_dir, "slope.tif")

    export_kwargs = {
        "region": region_aligned,
        "file_per_band": False,
        "crs": crs_str,
    }
    if crs_transform_list is not None:
        export_kwargs["crs_transform"] = crs_transform_list
    else:
        # fallback: use the pixel scale captured in grid (meters per pixel)
        export_kwargs["scale"] = float(grid["scale_m"])

    geemap.ee_export_image(slope_img, filename=slope_tif, **export_kwargs)

    # Wait briefly to ensure the file appears
    for _ in range(10):
        if os.path.exists(slope_tif):
            break
        time.sleep(0.5)
    if not os.path.exists(slope_tif):
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"Export failed, file not created: {slope_tif}")

    # 4) Read back as NumPy, convert nodata -> NaN, enforce DEM nodata mask
    try:
        with rasterio.open(slope_tif) as src:
            slope_np = src.read(1).astype("float32")
            nodata_val = src.nodata
    finally:
        # clean up temp dir
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if nodata_val is not None:
        slope_np = np.where(np.isclose(slope_np, nodata_val), np.nan, slope_np)

    # Enforce DEM nodata mask from grid
    nodata_mask = grid["nodata_mask"]
    slope_np = np.where(nodata_mask, np.nan, slope_np)

    # Sanity check: shapes must match DEM array
    dem_r = grid["dem"]
    if slope_np.shape != dem_r.shape:
        raise ValueError(f"Grid mismatch: slope {slope_np.shape} vs DEM {dem_r.shape}")

    return slope_np

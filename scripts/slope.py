import ee
import geemap
import numpy as np
import rasterio
import tempfile
import os
import time
import shutil
import io
import contextlib
import logging

def compute_slope(dem):
    """
    Computation of terrain slope based on DEM.
    """
    slope = ee.Terrain.slope(dem).rename("Slope")
    
    return slope

def slope_ee_to_numpy_on_grid(
    grid: dict,
    ee_dem_grid: ee.Image,
    *,
    quiet: bool = True,
) -> np.ndarray:
    """
    Convert EE slope (computed from ee_dem_grid) to a NumPy array on the SAME grid as DEM.
    Suppresses geemap/Google client prints when quiet=True.
    """
    # 1) Projection info from grid
    proj_info = grid["projection_info"]          # {'crs': str, 'transform': list|None}
    crs_str = proj_info["crs"]
    crs_transform_list = proj_info["transform"]
    region_aligned = grid["region_used"]

    # 2) Slope image on the aligned EE grid
    slope_img = ee.Terrain.slope(ee_dem_grid).toFloat().rename("slope")

    # 3) Export params (lock CRS + transform)
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
        export_kwargs["scale"] = float(grid["scale_m"])

    # Lower log verbosity if requested
    previous_levels = {}
    if quiet:
        for name in ("google", "googleapiclient", "geemap"):
            lg = logging.getLogger(name)
            previous_levels[name] = lg.level
            lg.setLevel(logging.ERROR)

    # Helper for silent export
    def _ee_export(img, filename, **kwargs):
        if quiet:
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                geemap.ee_export_image(img, filename=filename, **kwargs)
        else:
            geemap.ee_export_image(img, filename=filename, **kwargs)

    try:
        # Export (silent when quiet=True)
        _ee_export(slope_img, filename=slope_tif, **export_kwargs)

        # Krátké čekání, než se soubor objeví
        for _ in range(10):
            if os.path.exists(slope_tif):
                break
            time.sleep(0.5)
        if not os.path.exists(slope_tif):
            raise RuntimeError(f"Export failed, file not created: {slope_tif}")

        # 4) Read back → NumPy, převod NoData → NaN, aplikace DEM masky
        with rasterio.open(slope_tif) as src:
            slope_np = src.read(1).astype("float32")
            nodata_val = src.nodata
    finally:
        # Cleanup temp dir
        shutil.rmtree(tmp_dir, ignore_errors=True)
        # Restore logger levels
        if quiet:
            for name, lvl in previous_levels.items():
                logging.getLogger(name).setLevel(lvl)

    if nodata_val is not None:
        slope_np = np.where(np.isclose(slope_np, nodata_val), np.nan, slope_np)

    # Enforce DEM NoData mask
    slope_np = np.where(grid["nodata_mask"], np.nan, slope_np)

    # Shape check
    dem_r = grid["dem"]
    if slope_np.shape != dem_r.shape:
        raise ValueError(f"Grid mismatch: slope {slope_np.shape} vs DEM {dem_r.shape}")

    return slope_np

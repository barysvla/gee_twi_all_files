import ee
import geemap
import numpy as np
from pysheds.grid import Grid
import rasterio
import tempfile, os, shutil

def compute_flow_accumulation_pysheds(
    dem: ee.Image,
    scale: float,
    routing: str = 'mfd',    # 'd8' or 'mfd'
    area_units: str = 'km2', # 'cells' | 'm2' | 'km2'
    crs: str | None = None,  # optional EPSG like "EPSG:32633"; if None, use dem.projection()
    nodata_val: float = -9999.0
):
    """
    Compute flow accumulation using PySheds with proper area handling.

    - If area_units == 'cells': upstream cell counts (MFD -> fractional).
    - If area_units in {'m2','km2'}: weighted accumulation by per-pixel area (m²),
      yielding upstream area directly (m² or km²).
    """
    routing = routing.lower()
    if routing not in ('d8', 'mfd'):
        raise ValueError("routing must be 'd8' or 'mfd'")
    area_units = area_units.lower()
    if area_units not in ('cells', 'm2', 'km2'):
        raise ValueError("area_units must be 'cells', 'm2', or 'km2'")

    tmp_dir = tempfile.mkdtemp()
    dem_path = os.path.join(tmp_dir, "dem.tif")
    px_path  = os.path.join(tmp_dir, "pixel_area_m2.tif")

    try:
        # --- 1) Prepare EE images and export in the same grid ---
        dem_to_export = dem.unmask(nodata_val)
        px_img = ee.Image.pixelArea()
        if crs:
            px_img = px_img.reproject(crs=crs, scale=scale)
            export_kwargs = dict(scale=scale, region=dem.geometry(), file_per_band=False, crs=crs)
        else:
            px_img = px_img.reproject(dem.projection())
            export_kwargs = dict(scale=scale, region=dem.geometry(), file_per_band=False)

        geemap.ee_export_image(dem_to_export, filename=dem_path, **export_kwargs)
        geemap.ee_export_image(px_img,        filename=px_path,  **export_kwargs)

        # --- 2) Load as Raster objects (keep metadata and .nodata) ---
        grid   = Grid.from_raster(dem_path)
        dem_r  = grid.read_raster(dem_path)     # Raster (ndarray subclass with .nodata/.affine/.crs)
        px_r   = grid.read_raster(px_path)      # Raster; units are m² regardless of CRS in EE

        with rasterio.open(dem_path) as src:
            transform = src.transform
            crs_out   = src.crs

        # Determine DEM nodata (PySheds defaults to 0 if not present)
        nd = dem_r.nodata if getattr(dem_r, 'nodata', None) is not None else 0.0

        # --- 3) Conditioning on Raster (do NOT cast to ndarray) ---
        pit_filled = grid.fill_pits(dem_r, nodata=nd)
        flooded    = grid.fill_depressions(pit_filled, nodata=nd)
        inflated   = grid.resolve_flats(flooded, nodata=nd)

        # --- 4) Flow direction ---
        if routing == 'mfd':
            fdir = grid.flowdir(inflated, routing='mfd', apply_mask=True)
        else:
            fdir = grid.flowdir(inflated, apply_mask=True)  # D8

        # --- 5) Accumulation ---
        if area_units == 'cells':
            acc = grid.accumulation(fdir, routing=routing)
            acc_np = np.array(acc, dtype=np.float32)
        else:
            # Build a validity mask from DEM and zero-out weights where DEM is invalid.
            valid = np.isfinite(dem_r) & (dem_r != nd)
            px_r[~valid] = 0.0  # in-place, keeps px_r as Raster with .nodata

            # Weighted accumulation (sum of weights upstream) -> area in m²
            acc_m2 = grid.accumulation(fdir, routing=routing, weights=px_r)
            acc_np = np.array(acc_m2, dtype=np.float64)
            if area_units == 'km2':
                acc_np /= 1e6
            acc_np = acc_np.astype(np.float32)

            # Optional: mask invalids to NaN for downstream math/plots
            acc_np = np.where(valid, acc_np, np.nan).astype(np.float32)

        return acc_np, transform, crs_out

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

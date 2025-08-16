import ee
import geemap
import numpy as np
from pysheds.grid import Grid
import rasterio
import tempfile
import os
import shutil

def compute_flow_accumulation_pysheds(
    dem: ee.Image,
    scale: float,
    routing: str = 'mfd',   # 'd8' or 'mfd'
    area_units: str = 'km2' # 'cells' | 'm2' | 'km2'
):
    """
    Compute flow accumulation from an ee.Image DEM using PySheds.
    Returns a NumPy array aligned to the exported DEM, plus transform and CRS.

    Key behavior:
      - If area_units == 'cells': standard accumulation (count/fraction of cells).
      - If area_units in {'m2','km2'}: uses *weighted* accumulation with per-pixel area
        weights (m²), yielding upstream area directly (m² or km²).

    References:
      - PySheds accumulation weights: upstream sum when 'weights' is provided.
      - ee.Image.pixelArea(): per-pixel area in square meters (projection-independent).
    """
    routing = str(routing).lower()
    if routing not in ('d8', 'mfd'):
        raise ValueError("routing must be 'd8' or 'mfd'.")
    area_units = str(area_units).lower()
    if area_units not in ('cells', 'm2', 'km2'):
        raise ValueError("area_units must be 'cells', 'm2', or 'km2'.")

    tmp_dir = tempfile.mkdtemp()
    dem_path = os.path.join(tmp_dir, "dem.tif")
    pxarea_path = os.path.join(tmp_dir, "pixel_area_m2.tif")

    try:
        # --- 1) Prepare EE images (ensure a stable NoData and alignment) ---
        # Use a safe NoData for DEM export to avoid PySheds defaulting to 0.
        nodata_val = -9999.0
        dem_to_export = dem.unmask(nodata_val)

        # Reproject pixelArea to DEM projection to ensure grid alignment on export
        px_area_img = ee.Image.pixelArea().reproject(dem.projection())

        # --- 2) Export DEM and pixel-area rasters with identical region/scale ---
        # Region: use DEM geometry; Scale: caller-provided metric resolution
        geemap.ee_export_image(
            dem_to_export, filename=dem_path, scale=scale,
            region=dem.geometry(), file_per_band=False
        )
        geemap.ee_export_image(
            px_area_img, filename=pxarea_path, scale=scale,
            region=dem.geometry(), file_per_band=False
        )

        # --- 3) Load DEM as base grid; read metadata ---
        grid = Grid.from_raster(dem_path)
        dem_np = grid.read_raster(dem_path).astype(np.float32)

        with rasterio.open(dem_path) as src:
            transform = src.transform
            crs = src.crs
            # Trust the sentinel we wrote:
            # If the exporter didn't write NoData, we still use nodata_val consistently.
            # (Rasterio may show nodata=None; we control masking below.)

        # Build a boolean mask of valid DEM cells
        valid = np.isfinite(dem_np) & (dem_np != nodata_val)

        # Optional: replace invalid cells with NaN for conditioning steps
        # (PySheds handles arrays; invalids won't contribute if kept masked via weights)
        dem_np = np.where(valid, dem_np, np.nan).astype(np.float32)

        # --- 4) DEM conditioning (fill pits, depressions, resolve flats) ---
        # See PySheds docs for recommended sequence.
        # pit_filled = grid.fill_pits(dem_np)
        flooded = grid.fill_depressions(dem_np)
        inflated = grid.resolve_flats(flooded)

        # --- 5) Flow direction ---
        if routing == 'mfd':
            fdir = grid.flowdir(inflated, routing='mfd')
        else:  # 'd8'
            fdir = grid.flowdir(inflated)  # default D8

        # --- 6) Flow accumulation ---
        if area_units == 'cells':
            # Standard accumulation: counts (MFD returns fractional contributions)
            acc = grid.accumulation(fdir, routing=routing)
            acc_np = np.array(acc, dtype=np.float32)

        else:
            # Weighted accumulation: sum of pixel areas upstream
            # Load per-pixel area (m²) aligned to DEM grid
            px_m2 = grid.read_raster(pxarea_path).astype(np.float64)

            # Zero-out weights where DEM is invalid to prevent spurious upstream area
            px_m2 = np.where(valid, px_m2, 0.0)

            # Compute upstream *area* directly (m²)
            acc_area_m2 = grid.accumulation(fdir, routing=routing, weights=px_m2)

            if area_units == 'm2':
                acc_np = np.array(acc_area_m2, dtype=np.float32)
            else:  # 'km2'
                acc_np = (np.array(acc_area_m2, dtype=np.float64) / 1e6).astype(np.float32)

        # Ensure invalid cells are marked with NaN in output (optional but useful)
        acc_np = np.where(valid, acc_np, np.nan).astype(np.float32)

        return acc_np, transform, crs

    finally:
        # Always clean up temp files
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

import ee
import geemap
import numpy as np
from pysheds.grid import Grid
import rasterio
import tempfile
import os

def compute_flow_accumulation_pysheds(
    dem,
    scale=90,
    routing='mfd',
    area_units='km2'  # 'cells' | 'm2' | 'km2'
):
    """
    Compute flow accumulation from an ee.Image DEM using PySheds and return it as NumPy array.

    Parameters:
        dem (ee.Image): Input DEM.
        scale (int): Export resolution in meters.
        routing (str): 'd8' or 'mfd'.
        area_units (str): 'cells' (default), 'm2', 'km2'
            - 'cells'  : return number (or fraction) of upstream cells
            - 'm2'/'km2': multiply by pixel area from EE pixelArea()

    Returns:
        acc_np    (np.ndarray, float32)  – accumulation grid (units according to area_units)
        transform (rasterio.Affine)      – affine transform
        crs       (rasterio.crs.CRS)     – CRS of the raster
    """
    routing = str(routing).lower()
    if routing not in ('d8', 'mfd'):
        raise ValueError("routing must be 'd8' or 'mfd'.")
    area_units = str(area_units).lower()
    if area_units not in ('cells', 'm2', 'km2'):
        raise ValueError("area_units must be 'cells', 'm2', or 'km2'.")

    # 1) Create temporary folder and DEM file path
    tmp_dir = tempfile.mkdtemp()
    dem_path = os.path.join(tmp_dir, "dem.tif")

    # 2) Export DEM from GEE to GeoTIFF (region = exact DEM footprint)
    geemap.ee_export_image(
        dem,
        filename=dem_path,
        scale=scale,
        region=dem.geometry(),
        file_per_band=False
    )

    # 3) Load DEM into PySheds grid + read as NumPy array
    grid = Grid.from_raster(dem_path)
    dem_np = grid.read_raster(dem_path).astype(np.float32)

    with rasterio.open(dem_path) as src:
        transform = src.transform
        crs = src.crs
        nodata_val = src.nodata if src.nodata is not None else -9999.0

    # Replace NaN with nodata
    dem_np = np.nan_to_num(dem_np, nan=nodata_val).astype(np.float32)

    # 4) Hydrological conditioning
    flooded = grid.fill_depressions(dem_np)
    inflated = grid.resolve_flats(flooded)

    # 5) Flow direction & accumulation (PySheds output = number/fraction of cells)
    if routing == 'mfd':
        fdir = grid.flowdir(inflated, routing='mfd')
        acc_cells = grid.accumulation(fdir, routing='mfd').astype(np.float32)
    else:  # 'd8'
        fdir = grid.flowdir(inflated)  # default D8
        acc_cells = grid.accumulation(fdir).astype(np.float32)

    # 6) Option: convert to area using EE pixelArea (m²)
    if area_units in ('m2', 'km2'):
        # Pixel area (m²) – EE returns m² regardless of the projection
        px_area_img = ee.Image.pixelArea().clip(dem.geometry())
        px_area_np = geemap.ee_to_numpy(
            px_area_img, region=dem.geometry(), scale=scale, bands=['area']
        )
        px_area_np = np.squeeze(np.asarray(px_area_np)).astype(np.float32)
        if np.ma.isMaskedArray(px_area_np):
            px_area_np = px_area_np.filled(np.nan)

        # Ensure shapes match
        if px_area_np.shape != acc_cells.shape:
            raise RuntimeError(f"Pixel area shape {px_area_np.shape} != acc shape {acc_cells.shape}")

        # Multiply by pixel area to get m²
        acc_np = (acc_cells * px_area_np).astype(np.float32)
        if area_units == 'km2':
            acc_np = (acc_np / 1e6).astype(np.float32)  # convert m² to km²
    else:
        acc_np = acc_cells  # return number of cells

    return acc_np, transform, crs

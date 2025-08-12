import ee
import geemap
import numpy as np
from pysheds.grid import Grid
import rasterio
import tempfile
import os

def compute_flow_accumulation_pysheds(dem_img, scale=90):
    """
    Compute flow accumulation from an ee.Image DEM using PySheds.

    Parameters:
        dem_img (ee.Image): Input DEM as Earth Engine Image.
        scale (int): Resolution for export in meters.

    Returns:
        ee.Image: Flow accumulation as Earth Engine Image.
    """

    # 1) Temporary file for DEM
    tmp_dir = tempfile.mkdtemp()
    dem_path = os.path.join(tmp_dir, "dem.tif")

    # 2) Export DEM from GEE to GeoTIFF
    geemap.ee_export_image(
        dem_img,
        filename=dem_path,
        scale=scale,
        file_per_band=False
    )

    # 3) Load into PySheds grid
    grid = Grid.from_raster(dem_path)
    dem_np = grid.read_raster(dem_path).astype(np.float32)

    # Nodata value
    with rasterio.open(dem_path) as src:
        transform = src.transform
        crs = str(src.crs)
        nodata_val = src.nodata if src.nodata is not None else -9999.0

    # Replace NaN with nodata
    dem_np = np.nan_to_num(dem_np, nan=nodata_val)

    # 4) Hydrological conditioning
    flooded = grid.fill_depressions(dem_np)
    inflated = grid.resolve_flats(flooded)

    # 5) Flow direction and accumulation (default D8)
    fdir = grid.flowdir(inflated)
    acc = grid.accumulation(fdir)

    # 6) Convert back to ee.Image
    transform_tuple = (
        transform.a, transform.b, transform.c,
        transform.d, transform.e, transform.f
    )

    acc_img = geemap.numpy_to_ee(
        acc.astype(np.float32),
        transform=transform_tuple,
        crs=crs
    )

    return acc_img.rename('FlowAcc_D8')

# import ee

# def compute_flow_accumulation(dem):
#     """
#     Výpočet akumulace toku na základě DEM.
#     Zde lze implementovat vlastní metodu (např. D8 algoritmus).
#     """
#     # Zatím jen načítání existující vrstvy
#     dataset_MERIT = ee.Image('MERIT/Hydro/v1_0_1')
#     flowAccumulation_MERIT = dataset_MERIT.select('upa')
    
#     return flowAccumulation_MERIT.rename("Flow_Accumulation")

import ee
import tempfile
import rasterio
import numpy as np
from pysheds.grid import Grid

def compute_flow_accumulation(dem_image, geometry=None, scale=90, routing='d8'):
    """
    Compute flow accumulation from an Earth Engine DEM image using PySheds.

    Parameters
    ----------
    dem_image : ee.Image
        DEM as Earth Engine image (already preprocessed or clipped to AOI).
    geometry : ee.Geometry, optional
        Area of interest to clip before exporting.
    scale : int
        Resolution in meters for export from GEE.
    routing : str
        Flow routing algorithm ('d8' or 'mfd').

    Returns
    -------
    numpy.ndarray
        Flow accumulation array.
    """

    # 1) Export DEM from Earth Engine to a temporary GeoTIFF
    tmpfile = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
    path_tif = tmpfile.name
    tmpfile.close()

    url = dem_image.getDownloadURL({
        'scale': scale,
        'region': geometry or dem_image.geometry(),
        'format': 'GEO_TIFF'
    })

    import requests
    r = requests.get(url, stream=True)
    with open(path_tif, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    # 2) Read DEM with PySheds
    grid = Grid.from_raster(path_tif)
    dem = grid.read_raster(path_tif).astype(np.float32)

    # Set nodata if not defined
    with rasterio.open(path_tif) as src:
        nodata_value = src.nodata if src.nodata is not None else -9999.0

    # 3) Condition DEM (fill depressions, resolve flats)
    dem_filled = grid.fill_depressions(dem)
    dem_conditioned = grid.resolve_flats(dem_filled)

    # 4) Compute flow direction and accumulation
    if routing.lower() == 'd8':
        fdir = grid.flowdir(dem_conditioned)
        acc = grid.accumulation(fdir)
    elif routing.lower() == 'mfd':
        fdir = grid.flowdir(dem_conditioned, routing='mfd')
        acc = grid.accumulation(fdir, routing='mfd')
    else:
        raise ValueError("routing must be 'd8' or 'mfd'")

    return acc

# import ee
# import geemap
# import numpy as np
# import rasterio
# import tempfile, os, shutil

# def compute_slope(dem_img, region, *, crs=None, crs_transform=None, scale=None):
#     """
#     Compute slope (degrees) on the same grid as DEM export.
#     Supply either (crs + crs_transform) for exact grid lock, or fallback to scale.
#     Returns np.ndarray (float32).
#     """
#     slope_img = ee.Terrain.slope(dem_img).rename("Slope")  # °. (Výpočet na metrickém gridu dem_img)
#     tmp_dir = tempfile.mkdtemp()
#     slope_path = os.path.join(tmp_dir, "slope_deg.tif")
#     try:
#         export_kwargs = dict(region=region, file_per_band=False)
#         if crs_transform is not None:
#             export_kwargs.update(dict(crs=crs, crs_transform=crs_transform))
#         else:
#             export_kwargs.update(dict(scale=scale))

#         geemap.ee_export_image(slope_img, filename=slope_path, **export_kwargs)

#         with rasterio.open(slope_path) as src:
#             slope_np = src.read(1).astype(np.float32)
#         # Sanitizace
#         slope_np = np.where(np.isfinite(slope_np), slope_np, np.nan).astype(np.float32)
#         return slope_np
#     finally:
#         shutil.rmtree(tmp_dir, ignore_errors=True)

# V2
# import ee
# import geemap
# import numpy as np

# def compute_slope(dem, region, scale=90):
#     """
#     Výpočet sklonu terénu na základě DEM.
#     """
#     slope = ee.Terrain.slope(dem).rename("Slope")
#     # Geom for region in ee_to_numpy
#     #geom = slope.geometry() 
#     slope_array = geemap.ee_to_numpy(slope, region=region, bands=['Slope'], scale=90)
#     slope_array = np.squeeze(slope_array).astype(np.float64)
#     return slope_array

# V3
# import ee
# import geemap
# import numpy as np
# import rasterio
# import tempfile
# import os

# def compute_slope(dem_img, geometry=None, scale=90):
#     """
#     Compute slope in GEE (degrees), export to GeoTIFF, read back as NumPy array.
#     Returns:
#         slope_deg_np (np.ndarray, float32)
#     """
#     # Slope in GEE (degrees)
#     slope_img = ee.Terrain.slope(dem_img).rename("Slope_deg")
#     if geometry is not None:
#         slope_img = slope_img.clip(geometry)

#     # Export to temporary GeoTIFF
#     tmp_dir = tempfile.mkdtemp()
#     slope_path = os.path.join(tmp_dir, "slope_deg.tif")

#     geemap.ee_export_image(
#         slope_img,
#         filename=slope_path,
#         scale=scale,
#         file_per_band=False
#         # region=geometry  # volitelně můžeš explicitně nastavit region
#     )

#     # Read as NumPy
#     with rasterio.open(slope_path) as src:
#         slope_np = src.read(1).astype(np.float32)

    # # sanitize to finite values
    # slope_np = np.where(np.isfinite(slope_np), slope_np, np.nan).astype(np.float32)
    # return slope_np

import ee
import geemap
import numpy as np
import rasterio
import tempfile
import os
import shutil

def ee_image_to_numpy(
    img: ee.Image,
    region: ee.Geometry,
    *,
    crs=None,
    crs_transform=None,
    scale=None,
    bands: list = None,
    dtype: str = "float32"
) -> np.ndarray:
    """
    Export an EE Image (e.g. slope) to GeoTIFF via geemap, then read as numpy array.

    Parameters:
      - img: ee.Image to export
      - region: ee.Geometry region to export
      - crs, crs_transform: optional specification to lock to a grid
      - scale: fallback scale if crs_transform not provided
      - bands: list of band names to extract (if None, export all bands)
      - dtype: output numpy dtype

    Returns:
      - numpy array of shape (bands, rows, cols) if multiple bands, or (rows, cols) if single band
    """

    # Rename/select bands if needed
    if bands is not None:
        img = img.select(bands)

    # Use geemap to export as GeoTIFF
    tmp_dir = tempfile.mkdtemp()
    fname = "tmp_export.tif"
    path = os.path.join(tmp_dir, fname)

    export_kwargs = {"region": region}
    if crs_transform is not None and crs is not None:
        export_kwargs["crs"] = crs
        export_kwargs["crs_transform"] = crs_transform
    else:
        if scale is None:
            raise ValueError("Either (crs + crs_transform) or scale must be provided")
        export_kwargs["scale"] = scale

    # Use geemap to export
    geemap.ee_export_image(img, filename=path, **export_kwargs)

    # Read the exported TIFF
    try:
        with rasterio.open(path) as src:
            data = src.read().astype(dtype)  # shape: (bands, rows, cols)
            # Convert nodata to nan
            nodata = src.nodata
            if nodata is not None:
                # for each band
                mask = np.isclose(data, nodata)
                data = np.where(mask, np.nan, data)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # If one band only, squeeze out the first dimension
    if data.shape[0] == 1:
        data = data[0]

    return data


def compute_slope_numpy_from_ee(
    dem_img: ee.Image,
    region: ee.Geometry,
    *,
    crs=None,
    crs_transform=None,
    scale=None
) -> np.ndarray:
    """
    Compute slope in EE, export to numpy, return slope in degrees.

    Returns 2D numpy array (rows, cols).
    """
    slope_img = ee.Terrain.slope(dem_img).rename("slope")
    slope_np = ee_image_to_numpy(
        slope_img,
        region,
        crs=crs,
        crs_transform=crs_transform,
        scale=scale,
        bands=["slope"],
        dtype="float32"
    )
    return slope_np

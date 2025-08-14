import ee
import geemap

def compute_slope(dem, region, scale=90):
    """
    Výpočet sklonu terénu na základě DEM.
    """
    slope = ee.Terrain.slope(dem).rename("Slope")
    # Geom for region in ee_to_numpy
    #geom = slope.geometry() 
    slope_array = geemap.ee_to_numpy(slope, region=region, bands=['Slope'], scale=90)
    slope_array = np.squeeze(slope_array).astype(np.float64)
    return slope_array

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

    # sanitize to finite values
    slope_np = np.where(np.isfinite(slope_np), slope_np, np.nan).astype(np.float32)
    return slope_np



# import ee

# def compute_twi(flow_accumulation, slope):
#     """
#     Výpočet Topographic Wetness Index (TWI).
#     """
#     safe_slope = slope.where(slope.eq(0), 0.1)
#     tan_slope = safe_slope.divide(180).multiply(ee.Number(3.14159265359)).tan()
#     twi = flow_accumulation.divide(tan_slope).log().rename("TWI")
#     scaled_twi = twi.multiply(1e8).toInt().rename("TWI_scaled")
    

#     return scaled_twi

import ee
import numpy as np
import rasterio
import tempfile
import os

def compute_twi(acc_np, slope_deg_np, transform, crs,
                                 out_dir=None, out_name="twi_scaled.tif",
                                 scale_to_int=True, nodata_int=-2147483648):
    """
    Compute TWI in NumPy: TWI = ln( a / tan(beta) )
    - acc_np: D8 upstream cell counts (float32)
    - slope_deg_np: slope in degrees (float32), from GEE
    - transform: rasterio Affine taken from DEM
    - crs: rasterio CRS taken from DEM
    Writes GeoTIFF and returns (out_path, ee_image) where ee_image is created via ee.Image.loadGeoTIFF.
    """

    # Ensure finite arrays
    acc = np.where(np.isfinite(acc_np), acc_np, 0.0).astype(np.float32)
    slope_deg = np.where(np.isfinite(slope_deg_np), slope_deg_np, 0.0).astype(np.float32)

    # Cell metrics
    cellsize_x = float(transform.a)
    cellsize_y = float(abs(transform.e))
    cell_area = cellsize_x * cellsize_y

    # Avoid zeros
    acc_pos = np.maximum(acc, 1.0)
    slope_rad = np.deg2rad(np.maximum(slope_deg, 0.1))  # clamp to avoid tan(0)
    tan_beta = np.tan(slope_rad)

    # TWI
    twi = np.log((acc_pos * cell_area) / tan_beta).astype(np.float32)

    # Optional scaling to int (compat with your previous pipeline)
    if scale_to_int:
        twi_out = (twi * 1e8).astype(np.int32)
        dtype = "int32"
        nodata = nodata_int
    else:
        twi_out = twi
        dtype = "float32"
        nodata = np.nan

    # Output path
    if out_dir is None:
        out_dir = tempfile.mkdtemp()
    out_path = os.path.join(out_dir, out_name)

    # Write GeoTIFF
    profile = {
        "driver": "GTiff",
        "height": twi_out.shape[0],
        "width": twi_out.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "DEFLATE",
    }
    if dtype == "int32":
        profile["nodata"] = nodata

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(twi_out, 1)

    # Optional: load back to EE directly from local path? (Not supported)
    # Proper way in notebook for quick use: upload to GCS (US) and ee.Image.loadGeoTIFF('gs://...').
    # For convenience, return None as ee_image here; caller can load from GCS when uploaded.
    ee_image = None

    return out_path, ee_image

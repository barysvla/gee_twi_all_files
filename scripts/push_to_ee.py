import os
import time
import numpy as np
import ee

def push_array_to_ee_geotiff(
    arr,
    *,
    transform,
    crs,
    nodata_mask=None,
    bucket_name,
    project_id,
    band_name="acc",
    tmp_dir=None,
    object_prefix="twi_uploads",
    nodata_value=np.nan,
    dtype="float32",
    build_mask_from_nodata=True
):
    """
    Write a numpy array to GeoTIFF, upload to GCS, and load as ee.Image.

    Parameters
    ----------
    arr : 2D ndarray
        Accumulation grid (e.g., acc_cells or acc_km2).
    transform : affine.Affine
        Geotransform matching the DEM grid.
    crs : rasterio CRS
        CRS matching the DEM grid.
    nodata_mask : 2D bool or None
        True where cells are invalid. If None, inferred from ~np.isfinite(arr) when nodata_value is NaN.
    bucket_name : str
        GCS bucket (must be in US multi-region / US-CENTRAL1 / dual with US-CENTRAL1 for ee.Image.loadGeoTIFF).
    project_id : str
        GCP project ID for Storage client.
    band_name : str
        Name of the output band in ee.Image.
    tmp_dir : str or None
        Temp directory for writing the GeoTIFF.
    object_prefix : str
        Prefix/path in the bucket.
    nodata_value : float or np.nan
        NoData sentinel to write into the TIFF. If NaN, a mask is written instead.
    dtype : str
        Output GDAL dtype, typically "float32".
    build_mask_from_nodata : bool
        If True and nodata_value is NaN, write an explicit TIFF mask from nodata_mask.

    Returns
    -------
    out : dict
        {
          "image": ee.Image,          # loaded image (single band named `band_name`)
          "gs_uri": str,              # gs:// URI
          "local_path": str,          # local GeoTIFF path
          "bucket_object": str,       # object name in bucket
        }
    """
    import rasterio
    from google.cloud import storage

    # 0) Prepare temp path
    if tmp_dir is None:
        import tempfile
        tmp_dir = tempfile.mkdtemp()
    tstamp = int(time.time())
    tif_name = f"{band_name}_{tstamp}.tif"
    local_path = os.path.join(tmp_dir, tif_name)

    A = np.asarray(arr, dtype=np.float32 if dtype == "float32" else dtype)
    H, W = A.shape

    # 1) Prepare NoData/mask
    if nodata_mask is None:
        if np.isnan(nodata_value):
            nodata_mask = ~np.isfinite(A)
        else:
            nodata_mask = np.zeros_like(A, dtype=bool)
    else:
        nodata_mask = np.asarray(nodata_mask, dtype=bool)

    A_write = A.copy()
    # If a finite nodata_value is requested, inject it into nodata pixels
    if np.isfinite(nodata_value):
        A_write[nodata_mask] = float(nodata_value)

    # 2) Raster profile (tiled + LZW + float predictor)
    profile = {
        "driver": "GTiff",
        "height": H,
        "width":  W,
        "count":  1,
        "dtype":  A_write.dtype,
        "crs":    crs,
        "transform": transform,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "LZW",
        "predictor": 3,          # floating-point predictor for LZW (best for float data)
        "nodata": (None if np.isnan(nodata_value) else float(nodata_value)),
    }

    # 3) Write GeoTIFF
    with rasterio.open(local_path, "w", **profile) as dst:
        dst.write(A_write, 1)
        # If nodata is NaN, write an explicit mask (Alpha)
        if np.isnan(nodata_value) and build_mask_from_nodata:
            dst.write_mask((~nodata_mask).astype("uint8") * 255)

    # 4) Upload to GCS
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    # Optional sanity check of location (EE requires US/US-CENTRAL1/dual incl. US-CENTRAL1)
    # bucket.reload(); print("Bucket location:", bucket.location)
    object_name = f"{object_prefix}/{tif_name}"
    blob = bucket.blob(object_name)
    blob.upload_from_filename(local_path)
    gs_uri = f"gs://{bucket_name}/{object_name}"

    # 5) Load as ee.Image (single band, rename)
    # Requires ee.Initialize() to be done by the caller.
    ee_img = ee.Image.loadGeoTIFF(gs_uri).rename(band_name)

    return {
        "image": ee_img,
        "gs_uri": gs_uri,
        "local_path": local_path,
        "bucket_object": object_name,
    }

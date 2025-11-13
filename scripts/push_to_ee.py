import os
import time
import numpy as np
import ee

from google.api_core.exceptions import NotFound, Forbidden, Conflict, BadRequest

# Extra imports for COG writing and GCS upload
import rasterio
from google.cloud import storage


def _get_or_create_bucket(storage_client, bucket_name: str, project_id: str,
                          location: str = "US", storage_class: str = "STANDARD"):
    """
    Return an existing bucket or create it in the required location (US).
    Enforces a location compatible with ee.Image.loadGeoTIFF.
    """
    # Bucket names must be lowercase per GCS rules
    bucket_name = bucket_name.lower()

    try:
        # Requires storage.buckets.get permission
        bucket = storage_client.get_bucket(bucket_name)
        bucket.reload()
    except NotFound:
        # Create if missing
        bucket = storage_client.bucket(bucket_name)
        bucket.storage_class = storage_class
        # Optional but recommended: uniform bucket-level access
        try:
            bucket.iam_configuration.uniform_bucket_level_access_enabled = True
        except Exception:
            # Older client versions may not expose this property
            pass

        bucket = storage_client.create_bucket(bucket, project=project_id, location=location)
        bucket.reload()
    except Forbidden as e:
        raise RuntimeError(
            f"Forbidden to access bucket '{bucket_name}'. "
            f"Ensure your identity has bucket metadata read (storage.buckets.get) and create permissions."
        ) from e
    except BadRequest as e:
        raise RuntimeError(f"Bad request when accessing/creating bucket '{bucket_name}': {e}") from e
    except Conflict:
        # Rare race: created by someone else between get and create
        bucket = storage_client.get_bucket(bucket_name)
        bucket.reload()

    # Enforce EE-compatible location for loadGeoTIFF
    loc = (bucket.location or "").upper()
    # Accept US multi-region, US-CENTRAL1 region, or dual including US
    if not (loc == "US" or loc == "US-CENTRAL1" or "US" in loc):
        raise RuntimeError(
            f"Bucket '{bucket_name}' is in location '{bucket.location}'. "
            "ee.Image.loadGeoTIFF requires the US multi-region, a dual-region including US-CENTRAL1, "
            "or the US-CENTRAL1 region."
        )
    return bucket


def _write_cog_local(
    array_np: np.ndarray,
    transform,
    crs: str,
    out_path: str,
    nodata_value: float = -9999.0,
    blocksize: int = 512,
    compress: str = "LZW",
):
    """
    Write a single-band Cloud Optimized GeoTIFF (COG) to disk.

    This uses the GDAL COG driver through rasterio. It ensures:
      - float32 data type,
      - a proper NODATA value instead of NaN,
      - internal tiling and compression,
      - metadata/IFD early in the file (COG layout).
    """
    # Force float32
    arr = array_np.astype("float32", copy=False)

    # Replace NaN with the explicit nodata value
    if np.isnan(arr).any():
        arr = np.where(np.isnan(arr), nodata_value, arr).astype("float32", copy=False)

    # Base COG profile
    profile = {
        "driver": "COG",            # COG driver (requires GDAL with COG support)
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": nodata_value,
        "compress": compress,
        "blocksize": blocksize,
        "overview_resampling": "average",
        "BIGTIFF": "IF_SAFER",
    }

    # Open and write
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)


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
    nodata_value=-9999.0,
    dtype="float32",
    build_mask_from_nodata=True,  # kept for signature compatibility; used to derive mask if not provided
):
    """
    Write a numpy array as a Cloud Optimized GeoTIFF (COG), upload it to GCS,
    and load it as an ee.Image via ee.Image.loadGeoTIFF.

    Parameters
    ----------
    arr : np.ndarray
        2D numpy array to be stored.
    transform : affine.Affine
        Georeferencing transform for the raster.
    crs : str
        Coordinate reference system (e.g., "EPSG:32633").
    nodata_mask : np.ndarray or None
        Boolean mask where True marks nodata pixels. If None and build_mask_from_nodata
        is True, nodata is inferred from non-finite values in `arr`.
    bucket_name : str
        Name of the GCS bucket to upload to. Must be in EE-compatible location.
    project_id : str
        GCP project ID (also used by Earth Engine).
    band_name : str
        Name of the band in the resulting ee.Image.
    tmp_dir : str or None
        Local temporary directory for the GeoTIFF. If None, a temp directory is created.
    object_prefix : str
        Folder/prefix in the bucket for organizing uploads.
    nodata_value : float
        Explicit nodata value to be written into the COG. Must be a finite number.
    dtype : str
        Target numpy dtype (only "float32" makes sense here).
    build_mask_from_nodata : bool
        If True and nodata_mask is None, infer nodata from non-finite values in `arr`.

    Returns
    -------
    dict
        {
          "image": ee.Image,
          "gs_uri": str,
          "local_path": str,
          "bucket_object": str,
        }
    """
    # Ensure nodata_value is finite; if not, fall back to -9999.0
    if not np.isfinite(nodata_value):
        nodata_value = -9999.0

    # 0) Prepare temp directory and filename
    if tmp_dir is None:
        import tempfile
        tmp_dir = tempfile.mkdtemp()
    tstamp = int(time.time())
    tif_name = f"{band_name}_{tstamp}.tif"
    local_path = os.path.join(tmp_dir, tif_name)

    # 1) Prepare array and nodata mask
    A = np.asarray(arr, dtype=np.float32 if dtype == "float32" else dtype)
    if A.ndim != 2:
        raise ValueError("push_array_to_ee_geotiff expects a 2D array (single band).")

    if nodata_mask is None:
        if build_mask_from_nodata:
            # Derive mask from NaN / non-finite values
            nodata_mask = ~np.isfinite(A)
        else:
            nodata_mask = np.zeros_like(A, dtype=bool)
    else:
        nodata_mask = np.asarray(nodata_mask, dtype=bool)
        if nodata_mask.shape != A.shape:
            raise ValueError("nodata_mask must have the same shape as the input array.")

    # Apply nodata_value where mask is True
    A_masked = A.copy()
    A_masked[nodata_mask] = nodata_value

    # 2) Write local COG
    _write_cog_local(
        array_np=A_masked,
        transform=transform,
        crs=crs,
        out_path=local_path,
        nodata_value=nodata_value,
        blocksize=512,
        compress="LZW",
    )

    # 3) Upload to GCS (ensure bucket exists and is in a supported location)
    storage_client = storage.Client(project=project_id)
    bucket = _get_or_create_bucket(
        storage_client,
        bucket_name=bucket_name,
        project_id=project_id,
        location="US",           # enforce EE-compatible location
        storage_class="STANDARD",
    )

    object_name = f"{object_prefix}/{tif_name}"
    blob = bucket.blob(object_name)
    # Explicit content type helps EE detect it properly
    blob.upload_from_filename(local_path, content_type="image/tiff")
    gs_uri = f"gs://{bucket.name}/{object_name}"

    # 4) Load as ee.Image (caller must have ee.Initialize() done)
    #    Even though the file is COG, it is still a valid GeoTIFF, so loadGeoTIFF works.
    ee_img = ee.Image.loadGeoTIFF(gs_uri).rename(band_name).toFloat()

    return {
        "image": ee_img,
        "gs_uri": gs_uri,
        "local_path": local_path,
        "bucket_object": object_name,
    }

import os
import time
import numpy as np
import ee

from google.api_core.exceptions import NotFound, Forbidden, Conflict, BadRequest

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
            pass  # older client versions may not expose this property

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
    # Accept US multi-region, US-CENTRAL1 region, or dual including US-CENTRAL1
    if not (loc == "US" or loc == "US-CENTRAL1" or "US" in loc):
        raise RuntimeError(
            f"Bucket '{bucket_name}' is in location '{bucket.location}'. "
            "ee.Image.loadGeoTIFF requires the US multi-region, a dual-region including US-CENTRAL1, "
            "or the US-CENTRAL1 region."
        )
    return bucket


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

    Returns
    -------
    dict: {
      "image": ee.Image,
      "gs_uri": str,
      "local_path": str,
      "bucket_object": str,
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

    # 1) Prepare data and mask
    A = np.asarray(arr, dtype=np.float32 if dtype == "float32" else dtype)
    H, W = A.shape

    if nodata_mask is None:
        nodata_mask = ~np.isfinite(A) if np.isnan(nodata_value) else np.zeros_like(A, dtype=bool)
    else:
        nodata_mask = np.asarray(nodata_mask, dtype=bool)

    A_write = A.copy()
    if np.isfinite(nodata_value):
        A_write[nodata_mask] = float(nodata_value)

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
        "predictor": 3,
        "nodata": (None if np.isnan(nodata_value) else float(nodata_value)),
    }

    with rasterio.open(local_path, "w", **profile) as dst:
        dst.write(A_write, 1)
        if np.isnan(nodata_value) and build_mask_from_nodata:
            dst.write_mask((~nodata_mask).astype("uint8") * 255)

    # 2) Upload to GCS (ensure bucket exists and is in a supported location)
    storage_client = storage.Client(project=project_id)
    bucket = _get_or_create_bucket(
        storage_client,
        bucket_name=bucket_name,
        project_id=project_id,
        location="US",           # enforce EE-compatible location
        storage_class="STANDARD"
    )

    object_name = f"{object_prefix}/{tif_name}"
    blob = bucket.blob(object_name)
    blob.upload_from_filename(local_path)
    gs_uri = f"gs://{bucket.name}/{object_name}"

    # 3) Load as ee.Image (caller must have ee.Initialize() done)
    ee_img = ee.Image.loadGeoTIFF(gs_uri).rename(band_name)

    return {
        "image": ee_img,
        "gs_uri": gs_uri,
        "local_path": local_path,
        "bucket_object": object_name,
    }

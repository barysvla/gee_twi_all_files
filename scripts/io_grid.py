import ee
import geemap
import numpy as np
import rasterio
import tempfile
import os

def prepare_aligned_dem_and_pixelarea(
    dem_image: ee.Image,
    region_geom: ee.Geometry,
    *,
    resample_method: str = "bilinear",
    tmp_dir: str | None = None,
    dem_filename: str = "dem.tif",
    px_filename: str = "pixel_area.tif"
):
    """
    Export a DEM and pixel-area from Earth Engine on an identical, pixel-locked grid,
    read them back with rasterio, and return arrays + metadata ready for processing.

    Returns dict with:
        dem            : 2D float32 ndarray (NaN for NoData)
        pixel_area_m2  : 2D float32 ndarray (m^2 per pixel; masked to DEM footprint)
        transform      : affine.Affine
        crs            : rasterio CRS
        nodata_mask    : 2D bool ndarray (True where NoData)
        paths          : dict with local file paths
        tmp_dir        : temp directory used
    """
    # 0) Temp dir
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp()

    dem_path = os.path.join(tmp_dir, dem_filename)
    px_path  = os.path.join(tmp_dir, px_filename)

    # 1) Build export kwargs from the DEM projection
    proj_info = dem_image.projection().getInfo()
    crs = proj_info["crs"]
    crs_transform = proj_info.get("transform")

    export_kwargs = {
        "region": region_geom,
        "file_per_band": False,
        "crs": crs,
    }
    if crs_transform:
        export_kwargs["crs_transform"] = crs_transform
    else:
        # Fallback: use nominal scale if no explicit transform is present
        scale = dem_image.projection().nominalScale().getInfo()
        export_kwargs["scale"] = float(scale)

    # 2) Prepare images for export
    dem_to_export = dem_image.resample(resample_method).toFloat().clip(region_geom)
    px_img = ee.Image.pixelArea().updateMask(dem_image.mask())  # align footprint to DEM

    # 3) Exports (pixel-perfect aligned by crs+crs_transform or by crs+scale)
    geemap.ee_export_image(dem_to_export, filename=dem_path, **export_kwargs)
    geemap.ee_export_image(px_img,        filename=px_path,  **export_kwargs)

    # 4) Read back with rasterio
    with rasterio.open(dem_path) as src:
        dem_ma   = src.read(1, masked=True).astype("float32")  # MaskedArray (NoData preserved)
        transform = src.transform
        out_crs   = src.crs

    # unify to NaN for computations
    dem = dem_ma.filled(np.nan).astype("float32")
    nodata_mask = ~np.isfinite(dem)

    with rasterio.open(px_path) as src_px:
        px = src_px.read(1).astype("float32")
        # strict alignment checks
        if (src_px.transform != transform) or (src_px.crs != out_crs) \
           or (src_px.width != dem.shape[1]) or (src_px.height != dem.shape[0]):
            raise ValueError("pixel_area is not aligned with DEM (transform/CRS/shape mismatch).")

    return {
        "dem": dem,
        "pixel_area_m2": px,
        "transform": transform,
        "crs": out_crs,
        "nodata_mask": nodata_mask,
        "paths": {"dem": dem_path, "pixel_area": px_path},
        "tmp_dir": tmp_dir,
    }

import ee
import geemap
import numpy as np
import rasterio
import tempfile
import os

def export_dem_and_area_to_arrays(
    src,                         # ee.ImageCollection | ee.Image | asset-id (str)
    region_geom: ee.Geometry,
    *,
    band: str | None = None,     # e.g., 'DEM' for Copernicus; FABDEM is single-band -> None
    resample_method: str = "bilinear",  # 'nearest' for crisp edges; 'bilinear' for smooth DEM
    nodata_value: float = -9999.0,
    snap_region_to_grid: bool = True,
    tmp_dir: str | None = None,
    dem_filename: str = "dem.tif",
    px_filename: str  = "pixel_area.tif",
):
    """
    Build mosaic (if needed), fix projection, align region, export DEM+pixelArea on identical grid,
    read back as numpy, and also return aligned ee.Images on the same grid.

    Returns dict with:
        dem               : (H,W) float64 ndarray, NaN = NoData
        pixel_area_m2     : (H,W) float64 ndarray
        transform         : rasterio Affine
        crs               : rasterio CRS
        nodata_mask       : (H,W) bool
        nd_value          : float
        projection_info   : {'crs': str, 'transform': list|None}
        scale_m           : float | None
        region_used       : ee.Geometry (aligned region)
        ee_dem_grid       : ee.Image (masked DEM, reprojected to the same grid)
        ee_px_area_grid   : ee.Image (pixelArea on the same grid, masked to DEM)
        paths             : {'dem': path, 'pixel_area': path}
        tmp_dir           : temp folder
    """
    # 0) Input -> ee.Image with stable projection (mosaic if collection)
    if isinstance(src, ee.image.Image):
        img = src if band is None else src.select([band])
        seed = img
    else:
        ic = ee.ImageCollection(src) if isinstance(src, str) else src
        if band is not None:
            ic = ic.select([band])
        # Optional: define ordering explicitly if needed (default mosaic is last-to-first)
        # ic = ic.sort('system:time_start')  # uncomment if you prefer time-ordering
        seed = ic.first()
        img  = ic.filterBounds(region_geom).mosaic()  # masked mosaic; masked pixels remain masked. :contentReference[oaicite:1]{index=1}

    proj = seed.projection()
    # Set default projection so downstream ops have a valid default (avoid reduceResolution errors). :contentReference[oaicite:2]{index=2}
    img  = img.setDefaultProjection(proj)

    # 1) Optionally snap region to the DEM grid
    if snap_region_to_grid:
        mask = ee.Image.constant(1).reproject(proj).clip(region_geom).selfMask()
        g = mask.geometry().transform(proj=proj, maxError=1)
        region_aligned = g.bounds(maxError=1, proj=proj)
    else:
        region_aligned = region_geom

    # 2) Export params: prefer explicit crs+crsTransform (locks pixel origin/size). :contentReference[oaicite:3]{index=3}
    proj_info = proj.getInfo()
    crs = proj_info["crs"]
    crs_transform = proj_info.get("transform", None)

    export_kwargs = {"region": region_aligned, "file_per_band": False, "crs": crs}
    scale_m = None
    if crs_transform:
        export_kwargs["crs_transform"] = crs_transform
    else:
        scale_m = ee.Image(img).projection().nominalScale().getInfo()
        export_kwargs["scale"] = float(scale_m)

    # 3) Resampling choice
    rm = (resample_method or "").lower()
    if rm in ("bilinear", "bicubic"):
        img_rs = ee.Image(img).resample(rm)  # only these two are valid for resample() in EE
    elif rm in ("nearest", "", None):
        img_rs = ee.Image(img)               # default NN; do not call resample()
    else:
        raise ValueError(f"Invalid resample_method: {resample_method}. Use 'bilinear', 'bicubic', or 'nearest'.")

    # ---- Aligned EE images on the exact same grid ----
    # Force the exact grid using reproject(crs, crsTransform); keep mask for slope. :contentReference[oaicite:4]{index=4}
    ee_dem_grid = (
        ee.Image(img_rs)
        .toDouble()
        .reproject(crs=crs, crsTransform=crs_transform)
        .clip(region_aligned)
        .updateMask(ee.Image(img).mask())
    )
    ee_px_area_grid = (
        ee.Image.pixelArea()                            # m² per pixel
        .reproject(crs=crs, crsTransform=crs_transform)  # lock to same grid
        .updateMask(ee.Image(img).mask())
        .clip(region_aligned)
    )  # :contentReference[oaicite:5]{index=5}

    # TIFF export uses an unmasked DEM with a stable NoData fill
    dem_for_export = ee_dem_grid.unmask(nodata_value)

    # 4) Temp files
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp()
    dem_path = os.path.join(tmp_dir, dem_filename)
    px_path  = os.path.join(tmp_dir, px_filename)

    # 5) Exports (pixel-perfect, same grid)
    geemap.ee_export_image(dem_for_export, filename=dem_path, **export_kwargs)
    geemap.ee_export_image(ee_px_area_grid, filename=px_path, **export_kwargs)

    # 6) Read with rasterio
    with rasterio.open(dem_path) as src_dem:
        dem_band  = src_dem.read(1).astype("float64")
        transform = src_dem.transform
        out_crs   = src_dem.crs
        nd_src    = src_dem.nodata

    with rasterio.open(px_path) as src_px:
        px = src_px.read(1).astype("float64")
        if (src_px.transform != transform) or (src_px.crs != out_crs) or \
           (src_px.width != dem_band.shape[1]) or (src_px.height != dem_band.shape[0]):
            raise ValueError("pixel_area is not aligned with DEM (transform/CRS/shape mismatch).")

    # 7) Harmonize NoData in-memory (NaN)
    nd_value = nd_src if nd_src is not None else float(nodata_value)
    nodata_mask = (dem_band == nd_value) | ~np.isfinite(dem_band)
    dem = dem_band.copy()
    dem[nodata_mask] = np.nan

    return {
        "dem": dem,
        "pixel_area_m2": px,
        "transform": transform,
        "crs": out_crs,
        "nodata_mask": nodata_mask,
        "nd_value": nd_value,
        "projection_info": {"crs": crs, "transform": crs_transform},
        "scale_m": scale_m,
        "region_used": region_aligned,
        "ee_dem_grid": ee_dem_grid,           # <<< aligned ee.Image (masked)
        "ee_px_area_grid": ee_px_area_grid,   # <<< aligned ee.Image (pixelArea)
        "paths": {"dem": dem_path, "pixel_area": px_path},
        "tmp_dir": tmp_dir,
    }

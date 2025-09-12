import ee
import geemap
import numpy as np
import rasterio
import tempfile
import os

def export_dem_and_area_to_arrays(
    src,                         # ee.ImageCollection | ee.Image | asset-id (str)
    region_geom: ee.Geometry,    # vstupní zájmové území (ne nutně zarovnané)
    *,
    band: str | None = None,     # např. 'DEM' pro Copernicus; FABDEM je single-band -> None
    resample_method: str = "bilinear",  # 'nearest' pro ostré okraje; 'bilinear' pro plynulý DEM
    nodata_value: float = -9999.0,
    snap_region_to_grid: bool = True,   # zarovnání regionu na mřížku DEM (doporučeno)
    tmp_dir: str | None = None,
    dem_filename: str = "dem.tif",
    px_filename: str  = "pixel_area.tif",
):
    """Build mosaic (if needed), fix projection, align region, export DEM+pixelArea on identical grid, read back.

    Returns dict with:
        dem               : (H,W) float64 ndarray, NaN = NoData
        pixel_area_m2     : (H,W) float64 ndarray
        transform         : rasterio Affine
        crs               : rasterio CRS
        nodata_mask       : (H,W) bool
        nd_value          : float (NoData uložené v TIFF)
        projection_info   : {'crs': str, 'transform': list|None}
        scale_m           : float | None (pokud nebyl k dispozici transform)
        region_used       : ee.Geometry (region použitý pro export)
        paths             : {'dem': path, 'pixel_area': path}
        tmp_dir           : temp folder
    """
    # --- 0) Vstup -> ee.Image s pevnou projekcí ---
    if isinstance(src, ee.image.Image):
        img = src if band is None else src.select([band])
        seed = img
    else:
        ic = ee.ImageCollection(src) if isinstance(src, str) else src
        if band is not None:
            ic = ic.select([band])
        seed = ic.first()                      # seed pro projekci
        img  = (ic.filterBounds(region_geom)
                  .mosaic())                   # mozaika z kolekce (mask-based)  # :contentReference[oaicite:4]{index=4}

    proj = seed.projection()
    # Mozaice nastav nativní projekci seed dlaždice (jinak by měla WGS84/1°).  # :contentReference[oaicite:5]{index=5}
    img  = img.setDefaultProjection(proj)      # :contentReference[oaicite:6]{index=6}

    # --- 1) Volitelné zarovnání regionu na mřížku DEM (bez chybného geometry(proj)) ---
    if snap_region_to_grid:
        mask = ee.Image.constant(1).reproject(proj).clip(region_geom).selfMask()
        g = mask.geometry()                                 # default EPSG:4326
        g = g.transform(proj=proj, maxError=1)              # NAMED args, bez ErrorMargin chyby  # :contentReference[oaicite:7]{index=7}
        region_aligned = g.bounds(maxError=1, proj=proj)
    else:
        region_aligned = region_geom

    # --- 2) Export parametry (preferuj crs+transform; fallback scale) ---
    proj_info = proj.getInfo()
    crs = proj_info["crs"]
    crs_transform = proj_info.get("transform", None)

    export_kwargs = {
        "region": region_aligned,
        "file_per_band": False,
        "crs": crs,
    }
    scale_m = None
    if crs_transform:
        export_kwargs["crs_transform"] = crs_transform
    else:
        scale_m = ee.Image(img).projection().nominalScale().getInfo()
        export_kwargs["scale"] = float(scale_m)

    # --- 3) Připrav EE snímky k exportu ---
    # Choose resampling properly:
    rm = (resample_method or "").lower()
    if rm in ("bilinear", "bicubic"):
         img_rs = ee.Image(img).resample(rm)   # only these two are valid
    elif rm in ("nearest", "", None):
          img_rs = ee.Image(img)                # default is nearest-neighbor; do NOT call resample()
    else:
         raise ValueError(f"Invalid resample_method: {resample_method}. Use 'bilinear', 'bicubic', or 'nearest'.")

    dem_for_export = (
         img_rs
        .toDouble()
        .unmask(nodata_value)                 # fill masked pixels with a stable NoData
    )

    # pixelArea v m² a maska DEMu -> identická footprint/GRID  # :contentReference[oaicite:8]{index=8}
    px_img = ee.Image.pixelArea().updateMask(ee.Image(img).mask())

    # --- 4) Dočasné soubory ---
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp()
    dem_path = os.path.join(tmp_dir, dem_filename)
    px_path  = os.path.join(tmp_dir, px_filename)

    # --- 5) Exporty (pixel-perfect grid) ---
    geemap.ee_export_image(dem_for_export, filename=dem_path, **export_kwargs)  # :contentReference[oaicite:9]{index=9}
    geemap.ee_export_image(px_img,        filename=px_path,  **export_kwargs)

    # --- 6) Načtení přes rasterio ---
    with rasterio.open(dem_path) as src_dem:
        dem_band  = src_dem.read(1).astype("float64")
        transform = src_dem.transform
        out_crs   = src_dem.crs
        nd_src    = src_dem.nodata  # může být None (ale my jsme NoData zapsali do pixelů)

    with rasterio.open(px_path) as src_px:
        px = src_px.read(1).astype("float64")
        # přísná kontrola zarovnání
        if (src_px.transform != transform) or (src_px.crs != out_crs) or \
           (src_px.width != dem_band.shape[1]) or (src_px.height != dem_band.shape[0]):
            raise ValueError("pixel_area is not aligned with DEM (transform/CRS/shape mismatch).")

    # --- 7) Sjednocení NoData (v paměti používej NaN) ---
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
        "paths": {"dem": dem_path, "pixel_area": px_path},
        "tmp_dir": tmp_dir,
    }

# import ee
# import geemap
# import numpy as np
# import rasterio
# import tempfile
# import os

# def prepare_aligned_dem_and_pixelarea(
#     dem_image: ee.Image,
#     region_geom: ee.Geometry,
#     *,
#     resample_method: str = "bilinear",
#     tmp_dir: str | None = None,
#     dem_filename: str = "dem.tif",
#     px_filename: str = "pixel_area.tif"
# ):
#     """
#     Export a DEM and pixel-area from Earth Engine on an identical, pixel-locked grid,
#     read them back with rasterio, and return arrays + metadata ready for processing.

#     Returns dict with:
#         dem            : 2D float32 ndarray (NaN for NoData)
#         pixel_area_m2  : 2D float32 ndarray (m^2 per pixel; masked to DEM footprint)
#         transform      : affine.Affine
#         crs            : rasterio CRS
#         nodata_mask    : 2D bool ndarray (True where NoData)
#         paths          : dict with local file paths
#         tmp_dir        : temp directory used
#     """
#     # 0) Temp dir
#     if tmp_dir is None:
#         tmp_dir = tempfile.mkdtemp()

#     dem_path = os.path.join(tmp_dir, dem_filename)
#     px_path  = os.path.join(tmp_dir, px_filename)

#     # 1) Build export kwargs from the DEM projection
#     proj_info = dem_image.projection().getInfo()
#     crs = proj_info["crs"]
#     crs_transform = proj_info.get("transform")

#     export_kwargs = {
#         "region": region_geom,
#         "file_per_band": False,
#         "crs": crs,
#     }
#     if crs_transform:
#         export_kwargs["crs_transform"] = crs_transform
#     else:
#         # Fallback: use nominal scale if no explicit transform is present
#         scale = dem_image.projection().nominalScale().getInfo()
#         export_kwargs["scale"] = float(scale)

#     # 2) Prepare images for export
#     dem_to_export = dem_image.resample(resample_method).toFloat().clip(region_geom)
#     px_img = ee.Image.pixelArea().updateMask(dem_image.mask())  # align footprint to DEM

#     # 3) Exports (pixel-perfect aligned by crs+crs_transform or by crs+scale)
#     geemap.ee_export_image(dem_to_export, filename=dem_path, **export_kwargs)
#     geemap.ee_export_image(px_img,        filename=px_path,  **export_kwargs)

#     # 4) Read back with rasterio
#     with rasterio.open(dem_path) as src:
#         dem_ma   = src.read(1, masked=True).astype("float32")  # MaskedArray (NoData preserved)
#         transform = src.transform
#         out_crs   = src.crs

#     # unify to NaN for computations
#     dem = dem_ma.filled(np.nan).astype("float32")
#     nodata_mask = ~np.isfinite(dem)

#     with rasterio.open(px_path) as src_px:
#         px = src_px.read(1).astype("float32")
#         # strict alignment checks
#         if (src_px.transform != transform) or (src_px.crs != out_crs) \
#            or (src_px.width != dem.shape[1]) or (src_px.height != dem.shape[0]):
#             raise ValueError("pixel_area is not aligned with DEM (transform/CRS/shape mismatch).")

#     return {
#         "dem": dem,
#         "pixel_area_m2": px,
#         "transform": transform,
#         "crs": out_crs,
#         "nodata_mask": nodata_mask,
#         "paths": {"dem": dem_path, "pixel_area": px_path},
#         "tmp_dir": tmp_dir,
#     }

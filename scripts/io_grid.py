"""
Export DEM and pixel area from Earth Engine to aligned GeoTIFFs and NumPy arrays,
without noisy console output by default.

- Accepts ee.ImageCollection / ee.Image / asset-id (str)
- Aligns region to the DEM grid (optional)
- Locks CRS + transform for pixel-perfect alignment
- Exports DEM (with stable NoData fill) and ee.Image.pixelArea() on the same grid
- Reads both with rasterio into NumPy arrays
"""

from __future__ import annotations

import io
import os
import ee
import geemap
import numpy as np
import rasterio
import tempfile
import logging
import contextlib
from typing import Any, Dict, Optional, Union


def export_dem_and_area_to_arrays(
    src: Union[ee.image.Image, ee.imagecollection.ImageCollection, str],
    region_geom: ee.Geometry,
    *,
    band: Optional[str] = None,                # e.g. 'DEM' for Copernicus; None for single-band
    resample_method: str = "bilinear",         # 'nearest' | 'bilinear' | 'bicubic'
    nodata_value: float = -9999.0,
    snap_region_to_grid: bool = True,
    tmp_dir: Optional[str] = None,
    dem_filename: str = "dem.tif",
    px_filename: str = "pixel_area.tif",
    quiet: bool = True,                        # suppress prints/logs from geemap / google clients
) -> Dict[str, Any]:
    """
    Build mosaic (if needed), fix projection, align region, export DEM+pixelArea on identical grid,
    read back as NumPy, and also return aligned ee.Images on the same grid.

    Returns:
        dict with:
            dem: (H, W) float64 ndarray, NaN = NoData
            pixel_area_m2: (H, W) float64 ndarray
            transform: rasterio Affine
            crs: rasterio CRS
            nodata_mask: (H, W) bool
            nd_value: float
            projection_info: {'crs': str, 'transform': list|None}
            scale_m: float | None
            region_used: ee.Geometry (aligned region)
            ee_dem_grid: ee.Image (masked, grid-locked)
            ee_px_area_grid: ee.Image (grid-locked)
            paths: {'dem': path, 'pixel_area': path}
            tmp_dir: temp folder actually used
    """
    # --- Verbosity control ----------------------------------------------------
    # Reduce client-side logging noise when quiet=True
    previous_levels = {}
    if quiet:
        for name in ("google", "googleapiclient", "geemap"):
            logger = logging.getLogger(name)
            previous_levels[name] = logger.level
            logger.setLevel(logging.ERROR)

    # Utility to run geemap export with stdout/stderr redirection when quiet
    def _ee_export(img: ee.Image, filename: str, **kwargs) -> None:
        if quiet:
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                geemap.ee_export_image(img, filename=filename, **kwargs)
        else:
            geemap.ee_export_image(img, filename=filename, **kwargs)

    try:
        # --- 0) Normalize input to ee.Image with a stable projection ----------
        if isinstance(src, ee.image.Image):
            img = src if band is None else src.select([band])
            seed = img
        else:
            ic = ee.ImageCollection(src) if isinstance(src, str) else src
            if band is not None:
                ic = ic.select([band])
            # Use the first image's projection as seed; mosaic for coverage
            seed = ic.first()
            img = ic.filterBounds(region_geom).mosaic()

        proj = ee.Image(seed).projection()

        # Ensure default projection is set to avoid downstream reduceResolution issues
        img = ee.Image(img).setDefaultProjection(proj)

        # --- 1) Optionally snap region to the DEM grid ------------------------
        if snap_region_to_grid:
            # Create a 1-mask in the seed projection and take its geometry/bounds
            mask = ee.Image.constant(1).reproject(proj).clip(region_geom).selfMask()
            g = mask.geometry().transform(proj=proj, maxError=1)
            region_aligned = g.bounds(maxError=1, proj=proj)
        else:
            region_aligned = region_geom

        # --- 2) Export parameters: lock CRS + transform when available --------
        proj_info = proj.getInfo()
        crs = proj_info["crs"]
        crs_transform = proj_info.get("transform", None)

        export_kwargs = {
            "region": region_aligned,
            "file_per_band": False,
            "crs": crs,
        }
        scale_m: Optional[float] = None
        if crs_transform:
            export_kwargs["crs_transform"] = crs_transform
        else:
            # Fall back to nominal scale when explicit transform is absent
            scale_m = float(ee.Image(img).projection().nominalScale().getInfo())
            export_kwargs["scale"] = scale_m

        # --- 3) Resampling choice ---------------------------------------------
        rm = (resample_method or "").lower()
        if rm in ("bilinear", "bicubic"):
            img_rs = ee.Image(img).resample(rm)
        elif rm in ("nearest", "", None):
            img_rs = ee.Image(img)  # default NN (do not call resample)
        else:
            raise ValueError(
                f"Invalid resample_method: {resample_method}. "
                f"Use 'nearest', 'bilinear', or 'bicubic'."
            )

        # --- 4) Grid-lock both DEM and pixelArea to the same target grid ------
        ee_dem_grid = (
            ee.Image(img_rs)
            .toDouble()
            .reproject(crs=crs, crsTransform=crs_transform)
            .clip(region_aligned)
            .updateMask(ee.Image(img).mask())
        )

        ee_px_area_grid = (
            ee.Image.pixelArea()
            .reproject(crs=crs, crsTransform=crs_transform)
            .updateMask(ee.Image(img).mask())
            .clip(region_aligned)
        )

        # TIFF export uses an unmasked DEM with a stable NoData fill
        dem_for_export = ee_dem_grid.unmask(nodata_value)

        # --- 5) Temp files ----------------------------------------------------
        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp()
        dem_path = os.path.join(tmp_dir, dem_filename)
        px_path = os.path.join(tmp_dir, px_filename)

        # --- 6) Server-side exports via geemap (quiet by default) -------------
        _ee_export(dem_for_export, filename=dem_path, **export_kwargs)
        _ee_export(ee_px_area_grid, filename=px_path, **export_kwargs)

        # --- 7) Read back with rasterio ---------------------------------------
        with rasterio.open(dem_path) as src_dem:
            dem_band = src_dem.read(1).astype("float64")
            transform = src_dem.transform
            out_crs = src_dem.crs
            nd_src = src_dem.nodata

        with rasterio.open(px_path) as src_px:
            px = src_px.read(1).astype("float64")
            if (
                src_px.transform != transform
                or src_px.crs != out_crs
                or src_px.width != dem_band.shape[1]
                or src_px.height != dem_band.shape[0]
            ):
                raise ValueError(
                    "pixel_area is not aligned with DEM (transform/CRS/shape mismatch)."
                )

        # --- 8) Harmonize NoData in memory (NaN) ------------------------------
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
            "ee_dem_grid": ee_dem_grid,
            "ee_px_area_grid": ee_px_area_grid,
            "paths": {"dem": dem_path, "pixel_area": px_path},
            "tmp_dir": tmp_dir,
        }

    finally:
        # Restore previous logger levels if we changed them
        if quiet:
            for name, lvl in previous_levels.items():
                logging.getLogger(name).setLevel(lvl)

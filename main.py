import ee
import geemap
import numpy as np
import tempfile
import os

from scripts.io_grid import export_dem_and_area_to_arrays

from scripts.fill_depressions import priority_flood_fill
from scripts.resolve_flats import resolve_flats_barnes_tie

from scripts.flow_direction_quinn_cit import compute_flow_direction_quinn_cit
from scripts.flow_direction_quinn_1991 import compute_flow_direction_quinn_1991
from scripts.flow_direction_sfd_inf import compute_flow_direction_sfd_inf
from scripts.flow_direction_dz_mfd import compute_flow_direction_dz_mfd

from scripts.flow_accumulation_quinn_cit import compute_flow_accumulation_quinn_cit
from scripts.flow_accumulation_sfd_inf import compute_flow_accumulation_sfd_inf
from scripts.flow_accumulation_quinn_1991 import compute_flow_accumulation_quinn_1991

from scripts.push_to_ee import push_array_to_ee_geotiff 

from scripts.slope import compute_slope

from scripts.twi import compute_twi

from scripts.visualization import visualize_map
from scripts.visualization import vis_2sigma

# !Inicializace GEE!
ee.Initialize(project = 'gee-project-twi')

# !Definice zájmového území!
geometry = ee.Geometry.Rectangle([14.2, 50.0, 14.6, 50.2])

# Získání středu polygonu a nastavení zoomu
center = geometry.centroid().coordinates().getInfo()

# Načtení DEM
# 1) Collection
# FABDEM 30m
dem_raw = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")

# Copernicus GLO-30
#dem_raw = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM')  # DSM, band='DEM'

# ALOS World 3D 30 m
#dem_raw = ee.ImageCollection("JAXA/ALOS/AW3D30/V4_1").select('DSM')  # DSM, band='DSM'

# 1) Image ----------------------------
# SRTM DEM 30m
#dem_raw = ee.Image('USGS/SRTMGL1_003').select('elevation')
# NASA SRTM 30m 
#dem_raw = ee.Image("NASA/NASADEM_HGT/001").select('elevation')
# ASTER 30m
#dem_raw = ee.Image("projects/sat-io/open-datasets/ASTER/GDEM").select('b1');

# SRTM 90m
#dem_raw = ee.Image('CGIAR/SRTM90_V4').select('elevation')
# MERIT 90m
#dem_raw = ee.Image("MERIT/Hydro/v1_0_1").select("elv")

grid = export_dem_and_area_to_arrays(
    src=dem_raw,
    region_geom=geometry,
    band=None,
    resample_method="bilinear", 
    nodata_value=-9999.0,
    snap_region_to_grid=True
)

dem_r        = grid["dem"]
ee_dem_grid  = grid["ee_dem_grid"]
px_area      = grid["pixel_area_m2"]
transform    = grid["transform"]
nodata_mask  = grid["nodata_mask"]
out_crs      = grid["crs"]

scale = ee.Number(ee_dem_grid.projection().nominalScale())
print('nominalScale [m]:', scale.getInfo())

# Hydro conditioning
dem_filled, depth = priority_flood_fill(dem_r, seed_internal_nodata_as_outlet=True, return_fill_depth=True)
dem_resolved, flatmask, labels, stats = resolve_flats_barnes_tie(
    dem_filled, nodata=np.nan, epsilon=2e-5, equal_tol=1e-3, lower_tol=0.0, treat_oob_as_lower=True,
    require_low_edge_only=True, force_all_flats=False, include_equal_ties=True
)

# Compute flow direction
#flow_direction = compute_flow_direction_sfd_inf(dem_resolved, transform, nodata_mask=nodata_mask)
flow_direction = compute_flow_direction_quinn_1991(dem_resolved, transform, p=1.0, nodata_mask=nodata_mask)
#flow_direction = compute_flow_direction_dz_mfd(dem_resolved, p=1.6, nodata_mask=nodata_mask)

# Compute flow accumulation
#acc_km2 = compute_flow_accumulation_sfd_inf(flow_direction, pixel_area_m2=px_area,
#                                  nodata_mask=nodata_mask, out='km2')

#acc_km2 = compute_flow_accumulation_quinn_cit(flow_direction, pixel_area_m2=px_area, nodata_mask=nodata_mask, out='km2')

#acc_km2 = compute_flow_accumulation_quinn_1991(flow_direction, pixel_area_m2=px_area, nodata_mask=nodata_mask, out='km2')

#acc_cells = compute_flow_accumulation_sfd_inf(flow_direction, nodata_mask=nodata_mask, out='cells')
#acc_cells = compute_flow_accumulation_quinn_cit(flow_direction, pixel_area_m2=None, nodata_mask=nodata_mask, out='cells')
acc_cells = compute_flow_accumulation_quinn_1991(flow_direction, nodata_mask=nodata_mask, out='cells')

# Push numpy array to ee.Image GeoTIFF
dict_acc = push_array_to_ee_geotiff(
    acc_cells,
    transform=transform,
    crs=grid["crs"],
    nodata_mask=nodata_mask,
    bucket_name=f"{'gee-project-twi'}-ee-uploads",
    project_id="gee-project-twi",
    band_name="flow_accumulation",
    tmp_dir=grid.get("tmp_dir", None),
    nodata_value=np.nan,
)

ee_flow_accumulation = dict_acc["image"]

# Compute slope
slope = compute_slope(ee_dem_grid)

# Compute TWI
twi = compute_twi(ee_flow_accumulation, slope)

# CTI
cti_ic = ee.ImageCollection("projects/sat-io/open-datasets/HYDROGRAPHY90/flow_index/cti")
cti = cti_ic.mosaic().toFloat().clip(geometry)

# Visualization
vis_twi = vis_2sigma(twi, "TWI_scaled", geometry, scale, k=2.0,
                    palette=["#ff0000","#ffa500","#ffff00","#90ee90","#0000ff"])

vis_cti = vis_2sigma(cti, "b1", geometry, scale, k=2.0,
                    palette=["#ff0000","#ffa500","#ffff00","#90ee90","#0000ff"])
# vis_params_twi = {
#     "min": -529168144.8390943,
#     "max": 2694030.111316502,
#     "opacity": 1,
#     "palette": ["#ff0000", "#ffa500", "#ffff00", "#90ee90", "#0000ff"]
# }

# vis_params_slope = {
#     "bands": ["Slope"],
#     "min": 0,
#     "max": 90,
#     "palette": ["yellow", "red"]
# }
# vis_params_dem = {
#     "bands": ["elv"],
#     "min": 0,
#     "max": 3000,
#     "palette": ["black", "white"]
# }

# Create the map
Map = visualize_map([
    #(twi, vis_params_twi, "TWI"),
    (ee_flow_accumulation, {}, "Flow accumulation"),
    (cti, vis_cti, "CTI (Hydrography90m)"),
    (twi, vis_twi, "TWI (2σ)")
    # (out.select("Slope"), vis_params_slope, "Slope"),
    # (out.select("elv"), vis_params_dem, "Elevation")
])

Map.setCenter(center[0], center[1], zoom=12)

import ee
import geemap
import numpy as np
import tempfile
import os

from scripts.io_grid import prepare_aligned_dem_and_pixelarea

from scripts.fill_depressions import priority_flood_fill
# from scripts.resolve_flats import resolve_flats_towards_lower_edge
# from scripts.resolve_flats import resolve_flats_towards_lower_edge_gm
from scripts.resolve_flats import resolve_flats_barnes

from scripts.flow_direction_quinn_cit import compute_flow_direction_quinn_cit
from scripts.flow_direction_quinn_1991 import compute_flow_direction_quinn_1991
from scripts.flow_direction_sfd_inf import compute_flow_direction_sfd_inf
from scripts.flow_direction_dz_mfd import compute_flow_direction_dz_mfd

from scripts.flow_accumulation_quinn_cit import compute_flow_accumulation_quinn_cit
from scripts.flow_accumulation_sfd_inf import compute_flow_accumulation_sfd_inf

from scripts.slope import compute_slope

from scripts.twi import compute_twi

# !Inicializace GEE!
ee.Initialize(project = 'gee-project-twi')

# !Definice zájmového území!
geometry = ee.Geometry.Rectangle([14.2, 50.0, 14.6, 50.2])

# Získání středu polygonu a nastavení zoomu
#center = geometry.centroid().coordinates().getInfo()

# Načtení DEM
dem_raw = ee.Image('CGIAR/SRTM90_V4').select('elevation')

# Right grid
grid = prepare_aligned_dem_and_pixelarea(dem_raw, geometry)

dem          = grid["dem"]
px_area      = grid["pixel_area_m2"]
transform    = grid["transform"]
nodata_mask  = grid["nodata_mask"]

# Hydro conditioning
dem_filled, depth = priority_flood_fill(dem, nodata=np.nan, seed_internal_nodata_as_outlet=True, return_fill_depth=True)
dem_out, flatmask, labels, stats = resolve_flats_barnes(
    dem_filled, nodata=np.nan, epsilon=2e-5, equal_tol=0.03, lower_tol=0.0,
    treat_oob_as_lower=True, require_low_edge_only=True, force_all_flats=False
)

# Compute flow direction
flow_sfd = compute_flow_direction_sfd_inf(dem_out, transform, nodata_mask=nodata_mask)

# Compute flow accumulation
acc_km2 = compute_flow_accumulation_sfd_inf(flow_sfd, pixel_area_m2=px_area,
                                  nodata_mask=nodata_mask, out='km2')

# acc_cells = compute_flow_accumulation_sfd_inf(flow_sfd, nodata_mask=nodata_mask, out='cells')

# Push numpy array to ee.Image GeoTIFF
dict_acc = push_array_to_ee_geotiff(
    acc_km2,
    transform=transform,
    crs=grid["crs"],
    nodata_mask=nodata_mask,
    bucket_name=f"{'gee-project-twi'}-ee-uploads",
    project_id="gee-project-twi",
    band_name="acc_km2",
    tmp_dir=grid.get("tmp_dir", None),
    nodata_value=np.nan,
)

ee_flow_accumulation = dict_acc["image"]

# Compute slope
slope = compute_slope(dem_raw)

twi = compute_twi(ee_flow_accumulation, slope)

# Visualization
vis_params_twi = {
    "bands": ["TWI_scaled"],
    "min": -529168144.8390943,
    "max": 2694030.111316502,
    "opacity": 1,
    "palette": ["#ff0000", "#ffa500", "#ffff00", "#90ee90", "#0000ff"]
}
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
    (twi, vis_params_twi, "TWI"),
    (ee_flow_accumulation, {}, "flow accumulation (km2)")
    # (out.select("Slope"), vis_params_slope, "Slope"),
    # (out.select("elv"), vis_params_dem, "Elevation")
])

Map.setCenter(center[0], center[1], zoom=12)

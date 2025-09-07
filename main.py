import ee
import geemap
import numpy as np
import tempfile
import os

from io_grid import prepare_aligned_dem_and_pixelarea

from fill_depressions import priority_flood_fill
# from resolve_flats import resolve_flats_towards_lower_edge
# from resolve_flats import resolve_flats_towards_lower_edge_gm
from resolve_flats import resolve_flats_barnes

from flow_direction_quinn_cit import compute_flow_direction_quinn_cit

from flow_accumulation_quinn_cit import compute_flow_accumulation_quinn_cit

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

dem_r        = grid["dem"]
px_r         = grid["pixel_area_m2"]
transform    = grid["transform"]
nodata_mask  = grid["nodata_mask"]

# Hydro conditioning
dem_filled, depth = priority_flood_fill(dem_r, nodata=np.nan, seed_internal_nodata_as_outlet=True, return_fill_depth=True)
dem_out, flatmask, labels, stats = resolve_flats_barnes(
    dem_filled, nodata=np.nan, epsilon=2e-5, equal_tol=0.03, lower_tol=0.0,
    treat_oob_as_lower=True, require_low_edge_only=True, force_all_flats=False
)

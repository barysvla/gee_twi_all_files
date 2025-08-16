import ee
import geemap
import numpy as np

# Import vlastních modulů
from scripts.flow_accumulation_hydro import compute_flow_accumulation_hydro
from scripts.flow_accumulation_pysheds import compute_flow_accumulation_pysheds
from scripts.slope import compute_slope
from scripts.slope_ee_image import compute_slope_ee_image
from scripts.twi import compute_twi
from scripts.twi_np import compute_twi_numpy
from scripts.twi_np import compute_twi_numpy_like_ee
from scripts.visualization import visualize_map
#from scripts.export import export_to_drive, export_to_asset

# !Inicializace GEE!
ee.Initialize(project = 'gee-project-twi')

# !Definice zájmového území!
geometry = ee.Geometry.Rectangle([14.2, 50.0, 14.6, 50.2])

# Získání středu polygonu a nastavení zoomu
#center = geometry.centroid().coordinates().getInfo()

# Načtení DEM
dataset = ee.Image("MERIT/Hydro/v1_0_1")
dem = dataset.select("elv").clip(geometry)

proj = dem.projection()
scale_m = proj.nominalScale().getInfo()

dem_fix = dem.reproject(crs="EPSG:32633", scale=scale_m)

# >>> ZÍSKÁNÍ GRIDU PRO EXPORT <<<
proj_info = dem_fix.projection().getInfo()
crs = proj_info['crs']                       # např. 'EPSG:32633'
crs_transform = proj_info['transform']       # [a, b, c, d, e, f] affine
region = geometry.bounds(1)                  # jistý obdélník

# 1) Flow accumulation v NumPy (PySheds)
acc_m2, transform, out_crs = compute_flow_accumulation_pysheds(
    dem_fix, 
    # scale nepoužijeme, když předáváme crs_transform
    scale=scale_m, 
    routing='mfd', 
    area_units='m2',          # <<< ZÁSADNÍ: m², ne km²
    crs=crs, 
    crs_transform=crs_transform, 
    region=region
)

# 2) Slope v GEE → export → NumPy
slope_np = compute_slope(
    dem_fix, 
    region=region, 
    crs=crs, 
    crs_transform=crs_transform
)

# 3) TWI v NumPy → GeoTIFF
#twi_scaled = compute_twi_numpy_like_ee(acc_m2, slope_np, scale_to_int=True)

# 2) Slope v GEE → export → NumPy (ve stupních)
#slope_np = compute_slope(dem_fix, geometry, scale=scale_m)

# 3) TWI v NumPy → GeoTIFF
#twi_scaled = compute_twi_numpy(acc_np, slope_np, acc_is_area=True)
#twi_scaled = compute_twi_numpy_like_ee(acc_np, slope_np, scale_to_int=True)

# # Výpočet jednotlivých vrstev
flow_accumulation_hydro = compute_flow_accumulation_hydro(dem)
slope = compute_slope_ee_image(dem)
twi_hydro = compute_twi(flow_accumulation_hydro, slope)

twi_hydro = geemap.ee_to_numpy(twi_hydro, region=geometry, bands=['TWI_scaled'], scale=90)
twi_hydro = np.squeeze(twi_hydro).astype(np.float64)

# Kombinace vrstev
#out = dem.addBands(twi) #.addBands(flow_accumulation).addBands(slope)

# # Vizualizace
# vis_params_twi = {
#     "bands": ["TWI_scaled"],
#     "min": -529168144.8390943,
#     "max": 2694030.111316502,
#     "opacity": 1,
#     "palette": ["#ff0000", "#ffa500", "#ffff00", "#90ee90", "#0000ff"]
# }
#vis_params_slope = {
#    "bands": ["Slope"],
#    "min": 0,
#    "max": 90,
#    "palette": ["yellow", "red"]
#}
#vis_params_dem = {
#    "bands": ["elv"],
#    "min": 0,
#    "max": 3000,
#    "palette": ["black", "white"]
#}

## Vytvoření mapy
# Map = visualize_map([
#     (twi_hydro, vis_params_twi, "TWI_merit_hydro"),
#     (twi_pysheds, vis_params_twi, "TWI_pysheds")#,
#    # (out.select("Slope"), vis_params_slope, "Slope"),
#    # (out.select("elv"), vis_params_dem, "Elevation")
# ])

# Map.setCenter(center[0], center[1], zoom=12)

# Ověření, zda mapa obsahuje vrstvy
#for layer in Map.layers:
#    print(f"\t{layer.name}")
    
# Export výsledků do Google Drive
# task_drive = ee.batch.Export.image.toDrive(
#     image=twi,
#     description="TWI_Export",
#     folder="GEE_Exports",  # Název složky v Google Drive
#     fileNamePrefix="TWI_result",
#     region=geometry,
#     scale=90,  # Rozlišení odpovídající DEM
#     maxPixels=1e13,
#     fileFormat="GeoTIFF"
# )

# task_drive.start()
# print("📤 Export do Google Drive zahájen! Sledujte průběh v GEE Tasks.")









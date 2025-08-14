import ee
import geemap

# Import vlastních modulů
from scripts.flow_accumulation_hydro import compute_flow_accumulation_hydro
from scripts.flow_accumulation_pysheds import compute_flow_accumulation_pysheds
from scripts.slope import compute_slope
from scripts.twi import compute_twi
from scripts.visualization import visualize_map
#from scripts.export import export_to_drive, export_to_asset

# !Inicializace GEE!
ee.Initialize(project = 'gee-project-twi')

# !Definice zájmového území!
geometry = ee.Geometry.Rectangle([14.2, 50.0, 14.6, 50.2])

# Získání středu polygonu a nastavení zoomu
center = geometry.centroid().coordinates().getInfo()

# Načtení DEM
dataset_MERIT = ee.Image("MERIT/Hydro/v1_0_1")
dem = dataset_MERIT.select("elv").clip(geometry).reproject('EPSG:4326', None, 90)

# 1) Flow accumulation v NumPy (PySheds)
acc_np, transform, crs = compute_flow_accumulation_pysheds(dem, scale=90)

# 2) Slope v GEE → export → NumPy (ve stupních)
slope_deg_np = compute_slope(dem, geometry, scale=90)

# 3) TWI v NumPy → GeoTIFF → (volitelně) zpět do GEE jako ee.Image
twi_tif_path, twi_img = compute_twi_numpy_to_geotiff(
    acc_np=acc_np,
    slope_deg_np=slope_deg_np,
    transform=transform,
    crs=crs,
    out_dir=None,            # None => tempdir
    out_name="twi_scaled.tif",
    scale_to_int=True,       # shoda s tvým pipeline (x1e8, int32)
)

# # Výpočet jednotlivých vrstev
# flow_accumulation_hydro = compute_flow_accumulation_hydro(dem)
# flow_accumulation_pysheds = compute_flow_accumulation_pysheds(dem)
# slope = compute_slope(dem)
# twi_hydro = compute_twi(flow_accumulation_hydro, slope)
# twi_pysheds = compute_twi(flow_accumulation_pysheds, slope)

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






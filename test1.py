import ee
import geemap

# Inicializace Earth Engine
ee.Initialize(project = 'gee-project-twi')

# Definování oblasti zájmu – Praha (10 × 10 km)
roi = ee.Geometry.Rectangle([14.35, 50.05, 14.55, 50.15])

# Načtení satelitního snímku Landsat 8 (poslední dostupný snímek)
image = ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA") \
    .filterBounds(roi) \
    .sort("system:time_start", False) \
    .first() \
    .clip(roi)

# Nastavení vizualizace pro RGB (červená, zelená, modrá)
vis_params = {
    "bands": ["B4", "B3", "B2"],  # True Color (Red, Green, Blue)
    "min": 0.02,
    "max": 0.3,
    "gamma": 1.4
}

# Vytvoření mapy
Map = geemap.Map()
#Map.centerObject(roi, 12)  # Přiblížení na Prahu
Map.addLayer(image, vis_params, "Landsat 8 True Color")
Map.addLayer(roi, {"color": "red"}, "Oblast zájmu")  # Přidáme červený obrys oblasti

# Zobrazení mapy
Map
import ee
import geemap
from IPython.display import display
#from google.cloud import secretmanager
from google.colab import userdata
import json
import os

# Import vlastních modulů
from scripts.flow_accumulation import compute_flow_accumulation
from scripts.slope import compute_slope
from scripts.twi import compute_twi
from scripts.visualization import visualize_map
#from scripts.export import export_to_drive, export_to_asset

#-------------------------------------------------
# Načtení klíče ze Secrets
service_key_json = os.environ.get('SERVICE_KEY')

if service_key_json is None:
    raise ValueError("❌ Service Account Key není dostupný! Zkontroluj Secrets v Colabu.")

# Uložení klíče do dočasného souboru
key_path = "/content/service-key.json"
with open(key_path, "w") as key_file:
    key_file.write(service_key_json)

# Přihlášení k Earth Engine
service_account = "gee-service-twi@gee-project-twi.iam.gserviceaccount.com"
credentials = ee.ServiceAccountCredentials(service_account, key_path)
ee.Initialize(credentials)

#-------------------------------------------------
# Autentizace a inicializace GEE
# 🔹 Získání klíče, pokud existuje
#service_key_json = userdata.get('SERVICE_KEY')

# 🔹 Uživatel musí ručně nastavit svůj klíč
#if "SERVICE_KEY" not in userdata:
#    raise ValueError("❌ Nebyl nalezen Service Key! Zadej ho do `userdata['SERVICE_KEY']`.")

# 🔹 Načtení JSON jako slovník
#service_account_info = json.loads(service_key_json)

# 🔹 Přihlášení do Google Earth Engine
#credentials = ee.ServiceAccountCredentials(service_account_info['client_email'], service_account_info)
#ee.Initialize(credentials)
#-------------------------------------------------
#def get_service_key():
#    """Načte service account key ze Secret Manageru."""
#    client = secretmanager.SecretManagerServiceClient()
#    name = f"projects/242376316640/secrets/SERVICE_KEY/versions/latest"
#    response = client.access_secret_version(name=name)
#    return json.loads(response.payload.data.decode("UTF-8"))

# Načtení klíče z Google Secret Manager
#service_key = get_service_key()

# Přihlášení do Earth Engine
#credentials = ee.ServiceAccountCredentials(service_key["client_email"], service_key)
#ee.Initialize(credentials)
#-------------------------------------------------
# Cesta k JSON klíči
#key_path = "/content/gee_twi/service-key.json"

# Přihlášení pomocí Service Account
#service_account = "gee-service-twi@gee-project-twi.iam.gserviceaccount.com"
#credentials = ee.ServiceAccountCredentials(service_account, key_path)
#ee.Initialize(credentials)
#-------------------------------------------------
#ee.Authenticate()
#ee.Initialize()
#project = 'gee-project-twi'
#-------------------------------------------------
# Definice oblasti zájmu (Praha)
geometry = ee.Geometry.Rectangle([14.2, 50.0, 14.6, 50.2])

# Načtení DEM
dataset_MERIT = ee.Image("MERIT/Hydro/v1_0_1")
dem = dataset_MERIT.select("elv").clip(geometry)

# Výpočet jednotlivých vrstev
flow_accumulation = compute_flow_accumulation(dem)
slope = compute_slope(dem)
twi = compute_twi(flow_accumulation, slope)

# Kombinace vrstev
out = dem.addBands(flow_accumulation).addBands(slope).addBands(twi)

# Vizualizace
vis_params_twi = {
    "bands": ["TWI_scaled"],
    "min": -529168144.8390943,
    "max": 2694030.111316502,
    "opacity": 1,
    "palette": ["#ff0000", "#ffa500", "#ffff00", "#90ee90", "#0000ff"]
}
vis_params_slope = {
    "bands": ["Slope"],
    "min": 0,
    "max": 90,
    "palette": ["yellow", "red"]
}
vis_params_dem = {
    "bands": ["elv"],
    "min": 0,
    "max": 3000,
    "palette": ["black", "white"]
}

# Vytvoření mapy
Map = visualize_map([
    (out.select("TWI_scaled"), vis_params_twi, "TWI"),
    (out.select("Slope"), vis_params_slope, "Slope"),
    (out.select("elv"), vis_params_dem, "Elevation")
])

# Ověření, zda mapa obsahuje vrstvy
for layer in Map.layers:
    print(layer.name)
    
# Export výsledků
#export_to_drive(twi, "TWI_Export", "TWI_result", geometry)
#export_to_asset(twi, "users/tvoje_jmeno/TWI_result", geometry)

# Zobrazení mapy v Google Colab
Map

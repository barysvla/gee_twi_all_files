import ee
import geemap
from IPython.display import display
#from google.cloud import secretmanager
from google.colab import userdata
import json

# Import vlastních modulů
from scripts.flow_accumulation import compute_flow_accumulation
from scripts.slope import compute_slope
from scripts.twi import compute_twi
from scripts.visualization import visualize_map
#from scripts.export import export_to_drive, export_to_asset

#-------------------------------------------------
# 🔹 Uložení service account klíče jako textové proměnné
service_key_json = """
{
  "type": "service_account",
  "project_id": "gee-project-twi",
  "private_key_id": "67f3ec4f29eb962ebdbc76c7ebd1a41e5087950a",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDufgcy1IINstaB\nCqbEQkpH48VFyHSh4sGG2S6Lhcss+puK7WH5I+vEoQ3Du2fwzxZ4ZVoqxWLivkCA\nLaPLHjwBt29v5J/Sn8kLVXldGzZM9xITCNnQ4x3+S1OHQXY2J7jlbyCQuSQL912+\n5G3OhB81blOx/bxA/zZel46ROTgviby6aF0+RLz/WYKPlBGTS1CkxAAlE0C2QVZp\ncTCBhA7P5ltDzINz58zOWrBknwUT6G15f/lu2GpaX6gvP0sz7e14ang48WTgdEJ8\nDXgJKUBM+TyYZchDogdR5FSxVHaUoiD/HJtqG+9eEgJdSqqxWVDEfnv0KIsbDFwW\nPDaNHX4tAgMBAAECggEAAotZDGZBHzOuqmvEGr3oZLQI+wzH8htn2WfKGR1Og2D7\nrSGfzf+iaUh7RjLSHQNaqUbEOiUyE4cxYeyw/ELvI6YSFFLzsQakJ3DByNXv9n8F\n2v43vsHBtSjZ3zvV8X7rQKAPyuwp2tnIho3qn4wOFHwm9vQn0VdwB3HhheH5N96B\n85zmLExu432osup41po/nQK4RTo64lR2W8KcCUMVlc/tvsgDaQrOQoldzkPXzVvy\n0odNSchjEPGub9CdXrkQ7So5ACE7IRR+WUzGoYD07zjzc7q+eiq1l9Lc+WvLhO+H\nW/dqRD5h6rzFpqBrxqBofyjodJaDT2W1n+Mh7GOxgQKBgQD7bYYlHD1rgxfwp+UJ\n/zSDoj2PV6agRSgwfNYIPaZ7Rxb7VNfkz4Y2hlTMQpiftD9JnW+//oigvd0l7g8g\ndNQo+i9pjCQqQpcVt8gU8vAHSUHAQ1q24M9+cPSyNG68sS7qun/DulYW3w72ehmk\nDcX/Kql65X7peCWQ+3S/ArH4zQKBgQDy1Ej+ZoPJi6gZ3n/SsuCUcufYF0hB/m5m\nNHZMfzd7O0T9e4hgjVCBnraxsSA1Ze2/+1C/yI95BH4F+yvxvikxym4ZGbunlSKN\nxoyzEzDM0wNx5QaxqdWRU+XvJa4HsJHodrK52emjK4RcY6KC43C+kXgs+PG4Tn1Z\n4+HqLSEa4QKBgQCCcD+34P68WDaDU5cvqIbGA9WMGB0J6NcA0ML5Y4KLUkZJ/apD\nuqYWg3pavfIQaKKsvlDLenHHcrjYhLi0TegDmkgeqeXZRtK7Ia1bsO112juSU18s\noVUEc/V+vlT077c3b6n5ESK4muBYXuAOjFa8GpXyfD1rQjm7DblznyJVwQKBgGKI\nc2oPbRB+Q+LjXxi/DtQN4DfWErL8rA164jneMUQm47K7yrXrAaznGxj1V61zQ8rw\nDm8T0ev4P67roYRBdnPGwXAb+gJhSJkg5PzRH68tPKudlF7JHGkREy88Kebi5gHf\nEFzLqLtWMCpRH6Ne6OPbIVmWVndEGic8ifI9B77hAoGAZfgVBYm5t9X5k6WVQ/Io\nhI/DiKMXdtq9kOkc8Kl98s4U/K/06pI8+iS7eoMNt1DT8l3apLseHBcC1vO0ttwx\nelWlKyZOzJxJxNimUT1KuQXXhZRZ6m7dBdBP/bmjJOvWTEEIwL8lSQo+5pzSJiVe\nduVSmjNiAK5eYunKSnFHuS8=\n-----END PRIVATE KEY-----\n",
  "client_email": "gee-service-twi@gee-project-twi.iam.gserviceaccount.com",
  "client_id": "104900013036195137165",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/gee-service-twi%40gee-project-twi.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}
"""

# 🔹 Převedení textového JSONu na Python objekt
service_account_info = json.loads(service_key_json)

# 🔹 Přihlášení do GEE
credentials = ee.ServiceAccountCredentials(service_account_info['client_email'], service_account_info)
ee.Initialize(credentials)

#-------------------------------------------------
# Autentizace a inicializace GEE
# 🔹 Získání klíče, pokud existuje
#service_key_json = userdata.get("SERVICE_KEY")

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
#ee.Initialize(project = 'gee-project-twi')
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

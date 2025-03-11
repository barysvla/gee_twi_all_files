import geemap
import ee
from IPython.display import display

def visualize_map(layers):
    """
    Vytvoření interaktivní mapy s vrstevnicemi.
    :param layers: Seznam tuple vrstvy a jejich vizualizačních parametrů.
    """
    Map = geemap.Map()
    
    for layer in layers:
        image, vis_params, name = layer
        if isinstance(image, ee.Image):  # Ověření, že je image správný typ
            Map.addLayer(image, vis_params, name)
        else:
            print(f"⚠ Varování: Vrstva '{name}' není typu ee.Image a nebude zobrazena!")
    
    return Map

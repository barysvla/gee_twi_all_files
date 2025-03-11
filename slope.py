import ee

def compute_slope(dem):
    """
    Výpočet sklonu terénu na základě DEM.
    """
    slope = ee.Terrain.slope(dem).rename("Slope")
    return slope

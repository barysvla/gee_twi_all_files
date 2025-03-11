import ee

def compute_flow_accumulation(dem):
    """
    Výpočet akumulace toku na základě DEM.
    Zde lze implementovat vlastní metodu (např. D8 algoritmus).
    """
    # Zatím jen načítání existující vrstvy
    dataset_MERIT = ee.Image('MERIT/Hydro/v1_0_1')
    flowAccumulation_MERIT = dataset_MERIT.select('upa')
    
    return flowAccumulation_MERIT.rename("Flow_Accumulation")

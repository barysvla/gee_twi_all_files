import ee

def compute_twi(flow_accumulation, slope):
    """
    Výpočet Topographic Wetness Index (TWI).
    """
    safe_slope = slope.where(slope.eq(0), 0.1)
    tan_slope = safe_slope.divide(180).multiply(ee.Number(3.14159265359)).tan()
    twi = flow_accumulation.divide(tan_slope).log().rename("TWI")
    #scaled_twi = twi.multiply(1e8).toInt().rename("TWI_scaled")

    return twi

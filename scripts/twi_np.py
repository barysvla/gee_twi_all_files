# import ee

# def compute_twi(flow_accumulation, slope):
#     """
#     Výpočet Topographic Wetness Index (TWI).
#     """
#     safe_slope = slope.where(slope.eq(0), 0.1)
#     tan_slope = safe_slope.divide(180).multiply(ee.Number(3.14159265359)).tan()
#     twi = flow_accumulation.divide(tan_slope).log().rename("TWI")
#     scaled_twi = twi.multiply(1e8).toInt().rename("TWI_scaled")

#     return scaled_twi

import numpy as np

def compute_twi_numpy(
    acc_np: np.ndarray,
    slope_deg_np: np.ndarray,
    *,
    acc_is_area: bool,           # True pokud acc_np je už plocha (m²); False pokud je to počet buněk
    cell_area: float | None = None,  # plocha jedné buňky v m² (nutné, pokud acc_is_area=False)
    min_slope_deg: float = 0.1,  # ochrana proti tan(0)
    scale_to_int: bool = True
):
    """
    TWI = ln( a / tan(beta) )
    a ........ upslope area [m²]
    beta ..... sklon v radiánech (z degrees)
    """
    # Sanitize vstupy
    acc = np.array(acc_np, dtype=np.float32)
    slope_deg = np.array(slope_deg_np, dtype=np.float32)

    # a) plocha povodí a [m²]
    if acc_is_area:
        a = np.where(np.isfinite(acc), acc, 0.0)
    else:
        if cell_area is None:
            raise ValueError("cell_area musí být zadána, pokud acc_np není plocha (m²).")
        a = np.where(np.isfinite(acc), acc * float(cell_area), 0.0)

    # b) tan(beta), beta z degrees + ochranný práh
    slope_deg_safe = np.maximum(slope_deg, float(min_slope_deg))
    tan_beta = np.tan(np.deg2rad(slope_deg_safe))
    tan_beta = np.where(np.isfinite(tan_beta) & (tan_beta > 0), tan_beta, 1e-3)

    # c) TWI
    a_safe = np.where(np.isfinite(a) & (a > 0), a, 1.0)
    twi = np.log(a_safe / tan_beta).astype(np.float32)

    twi_scaled = (twi * 1e8).astype(np.int32)
    return  twi_scaled
  


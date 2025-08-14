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
    acc = np.array(acc_np, dtype=np.float64)
    slope_deg = np.array(slope_deg_np, dtype=np.float64)

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
    twi = np.log(a_safe / tan_beta).astype(np.float64)

    twi_scaled = (twi * 1e8).astype(np.int32)
    return  twi_scaled


import numpy as np

def compute_twi_numpy_like_ee(acc_np: np.ndarray,
                                   slope_deg_np: np.ndarray,
                                   scale_to_int: bool = True,
                                   nodata_int: int = -2147483648):
    """
    TWI = ln( acc_area_m2 / tan(beta) )
    beta v °; 0° -> 0.1° (stejně jako v EE .where(slope.eq(0), 0.1)).
    Pokud scale_to_int=True: výstup je int32 = round(twi * 1e8), neplatné -> nodata_int.
    """
    acc = np.array(acc_np, dtype=np.float64)          # plocha [m²]
    slope_deg = np.array(slope_deg_np, dtype=np.float64)

    # 1) safe slope: 0 -> 0.1°
    safe_slope = np.where(slope_deg == 0.0, 0.1, slope_deg)

    # 2) tan(beta)
    tan_beta = np.tan(np.deg2rad(safe_slope))

    # 3) TWI = ln(acc / tan_beta); neplatné (acc<=0, tan<=0, NaN/Inf) -> NaN
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = acc / tan_beta
        ratio = np.where((acc > 0) & np.isfinite(ratio) & (ratio > 0), ratio, np.nan)
        twi = np.log(ratio).astype(np.float32)

    if not scale_to_int:
        return twi  # float32

    # 4) škálování jako v EE (×1e8) + nodata
    twi_scaled = np.full(twi.shape, nodata_int, dtype=np.int32)
    valid = np.isfinite(twi)
    twi_scaled[valid] = (twi[valid] * 1e8).astype(np.int32)
    return twi_scaled



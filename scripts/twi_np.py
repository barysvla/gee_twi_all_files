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

# ---------------------------------------------------------------------------------------------------------------
import numpy as np

def compute_twi_numpy(
    acc_np: np.ndarray,
    slope_deg_np: np.ndarray,
    *,
    acc_is_area: bool,
    cell_area: float = None,
    min_slope_deg: float = 0.1,
    nodata_mask: np.ndarray = None,
    out_dtype: str = "float32"
) -> np.ndarray:
    """
    Compute TWI (Topographic Wetness Index) from numpy arrays.

    Formula:
      TWI = ln( a / tan(beta) )
    where:
      - a = upslope contributing area [m²]
      - beta = slope angle in radians

    Parameters:
      - acc_np: numpy array of accumulation (either count of cells or area)
      - slope_deg_np: numpy array of slopes in degrees
      - acc_is_area: True if acc_np already gives area in m²; False if it is number of cells
      - cell_area: area of each cell in m² (required if acc_is_area=False)
      - min_slope_deg: minimum slope (in degrees) to avoid tan(0) or extremely steep slopes
      - nodata_mask: boolean mask array (True where nodata / invalid)
      - out_dtype: output dtype (e.g. "float32")

    Returns:
      - numpy array of TWI (float) with same shape as inputs
    """

    # Convert to float64 for stable computations
    acc = np.array(acc_np, dtype=np.float64)
    slope_deg = np.array(slope_deg_np, dtype=np.float64)

    # Handle nodata_mask: if provided, mask out those cells
    if nodata_mask is not None:
        nodata_mask = np.asarray(nodata_mask, dtype=bool)
    else:
        nodata_mask = np.zeros(acc.shape, dtype=bool)

    # Compute upslope area a in m²
    if acc_is_area:
        a = acc
    else:
        if cell_area is None:
            raise ValueError("cell_area must be provided if acc_is_area=False")
        a = acc * float(cell_area)

    # Replace non-finite or negative values in a with a small positive to avoid log zeros
    # But preserve nodata_mask separately
    a_safe = np.where((~nodata_mask) & (a > 0) & np.isfinite(a), a, np.nan)

    # Enforce minimum slope degree to avoid tan(0)
    slope_deg_safe = np.maximum(slope_deg, min_slope_deg)
    # Convert to radians
    slope_rad = np.deg2rad(slope_deg_safe)

    # Compute tangent
    tan_beta = np.tan(slope_rad)
    # Avoid zeros or negative (shouldn't be), set very low floor
    tan_beta = np.where((~nodata_mask) & (tan_beta > 0) & np.isfinite(tan_beta), tan_beta, 1e-6)

    # Compute TWI
    twi = np.log(a_safe / tan_beta)

    # Apply nodata mask: set to nan where nodata
    twi = np.where(nodata_mask, np.nan, twi)

    # Cast to output dtype
    twi = twi.astype(out_dtype)

    return twi

# ---------------------------------------------------------------------------------------------------------------
import numpy as np

# def compute_twi_numpy_early(
#     acc_np: np.ndarray,
#     slope_deg_np: np.ndarray,
#     *,
#     acc_is_area: bool,           # True pokud acc_np je už plocha (m²); False pokud je to počet buněk
#     cell_area: float | None = None,  # plocha jedné buňky v m² (nutné, pokud acc_is_area=False)
#     min_slope_deg: float = 0.1,  # ochrana proti tan(0)
#     scale_to_int: bool = True
# ):
#     """
#     TWI = ln( a / tan(beta) )
#     a ........ upslope area [m²]
#     beta ..... sklon v radiánech (z degrees)
#     """
#     # Sanitize vstupy
#     acc = np.array(acc_np, dtype=np.float64)
#     slope_deg = np.array(slope_deg_np, dtype=np.float64)

#     # a) plocha povodí a [m²]
#     if acc_is_area:
#         a = np.where(np.isfinite(acc), acc, 0.0)
#     else:
#         if cell_area is None:
#             raise ValueError("cell_area musí být zadána, pokud acc_np není plocha (m²).")
#         a = np.where(np.isfinite(acc), acc * float(cell_area), 0.0)

#     # b) tan(beta), beta z degrees + ochranný práh
#     slope_deg_safe = np.maximum(slope_deg, float(min_slope_deg))
#     tan_beta = np.tan(np.deg2rad(slope_deg_safe))
#     tan_beta = np.where(np.isfinite(tan_beta) & (tan_beta > 0), tan_beta, 1e-3)

#     # c) TWI
#     a_safe = np.where(np.isfinite(a) & (a > 0), a, 1.0)
#     twi = np.log(a_safe / tan_beta).astype(np.float64)

#     #twi_scaled = (twi * 1e8).astype(np.int32)
#     return  twi

# ---------------------------------------------------------------------------------------------------------------
# import numpy as np

# def compute_twi_numpy_like_ee(
#     acc_area_np: np.ndarray,   # upstream area A [m^2]
#     slope_deg_np: np.ndarray,     # slope in degrees (same grid)
#     *,
#     cellsize_m: float,            # grid cell length c [m] (e.g., abs(transform.a))
#     scale_to_int: bool = True,
#     nodata_int: int = -2147483648
# ) -> np.ndarray:
#     """
#     TWI = ln( (A / c) / tan(beta) ), where:
#       A .... contributing area [m^2]
#       c .... cell length [m]
#       beta . slope in degrees (0° clamped to 0.1°)
#     If scale_to_int=True, return int32 with EE-like scaling (×1e8) and nodata sentinel.
#     """
#     # -- 0) Inputs and basic checks
#     A = np.asarray(acc_area_np, dtype=np.float64)
#     slope_deg = np.asarray(slope_deg_np, dtype=np.float64)
#     if A.shape != slope_deg.shape:
#         raise ValueError(f"Shape mismatch: acc {A.shape} vs slope {slope_deg.shape}")
#     if not np.isfinite(cellsize_m) or cellsize_m <= 0:
#         raise ValueError("cellsize_m must be a positive finite value in meters.")

#     # -- 1) Specific catchment area a = A / c  [m^2/m]
#     sca = A / float(cellsize_m)

#     # -- 2) Safe slope: prevent tan(0)
#     safe_slope = np.where(slope_deg == 0.0, 0.1, slope_deg)
#     tan_beta = np.tan(np.deg2rad(safe_slope))

#     # -- 3) TWI; keep invalid cells as NaN
#     with np.errstate(divide='ignore', invalid='ignore'):
#         ratio = sca / tan_beta
#         valid_ratio = np.isfinite(ratio) & (ratio > 0) & np.isfinite(sca) & (sca > 0)
#         twi = np.full(A.shape, np.nan, dtype=np.float32)
#         twi[valid_ratio] = np.log(ratio[valid_ratio]).astype(np.float32)

#     if not scale_to_int:
#         return twi  # float32 result (NaN where invalid)

#     # -- 4) Robust scaling to int32
#     scaled = twi * 1e8  # float32
#     # Only keep values that are finite and inside int32 range when scaled
#     INT32_MIN, INT32_MAX = np.int32(-2147483648), np.int32(2147483647)
#     finite_mask = np.isfinite(scaled)
#     range_mask = (scaled >= INT32_MIN) & (scaled <= INT32_MAX)
#     vmask = finite_mask & range_mask

#     out = np.full(A.shape, nodata_int, dtype=np.int32)
#     # Use rounding before cast; mask guarantees no NaN/Inf, prevents cast warning
#     out[vmask] = np.rint(scaled[vmask]).astype(np.int32)
#     return out

# ---------------------------------------------------------------------------------------------------------------
# import numpy as np

# def compute_twi_numpy_like_ee(acc_np: np.ndarray,
#                                    slope_deg_np: np.ndarray,
#                                    scale_to_int: bool = True,
#                                    nodata_int: int = -2147483648):
#     """
#     TWI = ln( acc_area_m2 / tan(beta) )
#     beta v °; 0° -> 0.1° (stejně jako v EE .where(slope.eq(0), 0.1)).
#     Pokud scale_to_int=True: výstup je int32 = round(twi * 1e8), neplatné -> nodata_int.
#     """
#     acc = np.array(acc_np, dtype=np.float64)          # plocha [m²]
#     slope_deg = np.array(slope_deg_np, dtype=np.float64)

#     # 1) safe slope: 0 -> 0.1°
#     safe_slope = np.where(slope_deg == 0.0, 0.1, slope_deg)

#     # 2) tan(beta)
#     tan_beta = np.tan(np.deg2rad(safe_slope))

#     # 3) TWI = ln(acc / tan_beta); neplatné (acc<=0, tan<=0, NaN/Inf) -> NaN
#     with np.errstate(divide='ignore', invalid='ignore'):
#         ratio = acc / tan_beta
#         ratio = np.where((acc > 0) & np.isfinite(ratio) & (ratio > 0), ratio, np.nan)
#         twi = np.log(ratio).astype(np.float32)

#     if not scale_to_int:
#         return twi  # float32

#     # 4) škálování jako v EE (×1e8) + nodata
#     twi_scaled = np.full(twi.shape, nodata_int, dtype=np.int32)
#     valid = np.isfinite(twi)
#     twi_scaled[valid] = (twi[valid] * 1e8).astype(np.int32)
#     return twi_scaled








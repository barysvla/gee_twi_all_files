import numpy as np

# def compute_flow_direction_quinn_cit_simple(dem, scale=1.0, cit=1000.0, h=2.0):
#     """
#     Compute flow direction weights using Quinn 1995 with CIT (Channel Initiation Threshold).

#     Parameters:
#         dem (2D np.ndarray): elevation model
#         scale (float): resolution (cell size in meters)
#         cit (float): channel initiation threshold (in contributing area units)
#         h (float): exponent controlling p sensitivity to a (typically 1–2)

#     Returns:
#         flow_weights (3D np.ndarray): flow proportions to 8 neighbors in D8 order
#     """
#     rows, cols = dem.shape
#     flow = np.zeros((rows, cols, 8), dtype=np.float32)

#     D8 = [
#         (-1, 1),  # NE
#         (0, 1),   # E
#         (1, 1),   # SE
#         (1, 0),   # S
#         (1, -1),  # SW
#         (0, -1),  # W
#         (-1, -1), # NW
#         (-1, 0)   # N
#     ]
#     distances = np.array([np.sqrt(2), 1, np.sqrt(2), 1, np.sqrt(2), 1, np.sqrt(2), 1]) * scale
#     L = distances.copy()  # Assume L_i = d_i for simplification (can be modified)

#     # Initial contributing area per cell (in pixel units)
#     a = np.ones((rows, cols), dtype=np.float32)

#     for i in range(1, rows - 1):
#         for j in range(1, cols - 1):
#             center = dem[i, j]
#             slopes = []
#             valid_indices = []

#             for k, (dy, dx) in enumerate(D8):
#                 ni, nj = i + dy, j + dx
#                 dz = center - dem[ni, nj]
#                 tan_beta = dz / distances[k] if dz > 0 else 0
#                 if tan_beta > 0:
#                     slopes.append((L[k] * tan_beta))
#                     valid_indices.append(k)

#             if slopes:
#                 p = ((a[i, j] / cit) + 1) ** h
#                 powered = [s**p for s in slopes]
#                 total = sum(powered)
#                 if total > 0:
#                     for idx, k in enumerate(valid_indices):
#                         flow[i, j, k] = powered[idx] / total

#     return flow

# Varianta 2 ------------------------------------------------------------------------
import numpy as np

# ----------------------------- #
# 1) Pomůcky pro jednotky / vzdálenosti v EPSG:4326
# ----------------------------- #

def meters_per_degree_lat_lon(lat_deg):
    """
    Přesné (WGS-84) metry na 1° zem. šířky/délky v dané šířce.
    Vrací tuple (m_per_deg_lat, m_per_deg_lon), každé může být skalár nebo vektor.
    Zdroj: vztahy z poloměrů křivosti M a N na elipsoidu WGS-84. (viz citace)
    """
    # WGS-84
    a = 6378137.0                      # velká poloosa [m]
    f = 1.0 / 298.257223563
    e2 = (2.0 - f) * f                 # e^2

    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    s = np.sin(lat); c = np.cos(lat)
    one_minus_e2s2 = 1.0 - e2 * s*s

    # Poloměry křivosti
    M = a * (1.0 - e2) / (one_minus_e2s2 ** 1.5)  # meridionální
    N = a / np.sqrt(one_minus_e2s2)               # prime vertical

    m_per_deg_lat = (np.pi / 180.0) * M
    m_per_deg_lon = (np.pi / 180.0) * N * np.clip(c, 0.0, None)

    return m_per_deg_lat, m_per_deg_lon


def step_lengths_for_rows(transform, H):
    """
    Pro každý řádek DEM (index i) spočítá délky kroků v metrech:
      dx[i] = 1 pixel na východ/západ, dy[i] = 1 pixel na sever/jih
      ddiag[i] = diagonála sqrt(dx^2 + dy^2)
    Předpoklad: sever nahoře (běžné v EE/rasterio: b=d=0, a>0, e<0). 
    """
    # velikost pixelu ve stupních (absolutní)
    deg_x = float(abs(transform.a))
    deg_y = float(abs(transform.e))

    # zem. šířka středů řádků v °: y_center = f + e*(i+0.5) -> lat
    # když je transform bez rotace: x = a*col + c ; y = e*row + f
    lat0 = float(transform.f)
    step_lat = float(transform.e)  # záporné pro sever-nahoru
    lat_centers_deg = lat0 + step_lat * (np.arange(H, dtype=np.float64) + 0.5)

    m_per_deg_lat, m_per_deg_lon = meters_per_degree_lat_lon(lat_centers_deg)

    dx = m_per_deg_lon * deg_x
    dy = m_per_deg_lat * deg_y
    ddiag = np.hypot(dx, dy)

    return dx.astype(np.float64), dy.astype(np.float64), ddiag.astype(np.float64)


# ----------------------------- #
# 2) Flow direction (Quinn 1995 + CIT) pro EPSG:4326
# ----------------------------- #

# D8 sousedi v pořadí: NE, E, SE, S, SW, W, NW, N (pevné pythoní inty kvůli overflowu)
D8 = [(-1, 1), (0, 1), (1, 1), (1, 0),
      (1, -1), (0, -1), (-1, -1), (-1, 0)]

def compute_flow_direction_quinn_cit(
    dem, transform, *,
    cit=1000.0,          # práh pro CIT (ve stejných jednotkách jako "a0", viz area_unit)
    h=2.0,               # exponent p citlivý na přispívající plochu
    area_unit='cells',   # 'cells' nebo 'm2' (viz a0 níže)
    pixel_area_m2=None,  # nutné, pokud area_unit='m2'
    nodata_mask=None
):
    """
    Quinn 1995 (MFD) s CIT: váhy odtoku k 8 sousedům v EPSG:4326.
    Vzdálenosti mezi středy pixelů jsou v metrech (přes dx/dy/diag per-row).

    Návrat: flow (H, W, 8) float32, součet odtékajících vah ≤ 1.
    """
    Z = np.asarray(dem, dtype=np.float64)
    H, W = Z.shape

    if nodata_mask is None:
        nodata_mask = ~np.isfinite(Z)
    else:
        nodata_mask = np.asarray(nodata_mask, dtype=bool)

    # krokové délky v metrech pro každý řádek
    dx, dy, ddiag = step_lengths_for_rows(transform, H)

    # počáteční přispívající plocha a0 (pro CIT scale)
    if area_unit == 'cells':
        a0 = 1.0
        get_a0 = lambda i, j: 1.0
    elif area_unit == 'm2':
        if pixel_area_m2 is None:
            raise ValueError("pixel_area_m2 je povinné, pokud area_unit='m2'.")
        px = np.asarray(pixel_area_m2, dtype=np.float64)
        if px.shape != (H, W):
            raise ValueError("pixel_area_m2 musí mít tvar (H, W).")
        get_a0 = lambda i, j: px[i, j]
    else:
        raise ValueError("area_unit musí být 'cells' nebo 'm2'.")

    flow = np.zeros((H, W, 8), dtype=np.float32)

    # smyčka přes vnitřek; hranici ošetříme v podmínkách
    for i in range(H):
        for j in range(W):
            if nodata_mask[i, j]:
                continue

            zc = Z[i, j]

            # délky kroků z aktuální buňky (podle směru)
            step_len = (
                ddiag[i],  dx[i], ddiag[i], dy[i],  # NE, E, SE, S
                ddiag[i],  dx[i], ddiag[i], dy[i]   # SW, W, NW, N
            )

            slopes = []
            idxs = []

            # spočítej tan(beta) a směrový "S_i = L_i * tan(beta)"
            for k, (di, dj) in enumerate(D8):
                ni, nj = i + di, j + dj
                if not (0 <= ni < H and 0 <= nj < W):
                    continue
                if nodata_mask[ni, nj]:
                    continue

                dz = zc - Z[ni, nj]
                if dz <= 0:
                    continue

                d = step_len[k]
                if d <= 0:
                    continue

                tan_beta = dz / d
                S = d * tan_beta   # tj. S = dz  (ponechávám formulaci jako v tvém kódu)
                if S > 0:
                    slopes.append(S)
                    idxs.append(k)

            if not slopes:
                continue

            # Quinn CIT: p = ((a/cit) + 1)^h
            a = get_a0(i, j)
            p = ((a / float(cit)) + 1.0) ** float(h)

            powered = np.power(slopes, p, dtype=np.float64)
            total = float(np.sum(powered))
            if total <= 0.0:
                continue

            for s, k in zip(powered, idxs):
                flow[i, j, k] = np.float32(s / total)

    return flow

# import numpy as np

# # --- Pomůcky: metry na stupeň + délky kroků pro řádky (EPSG:4326) ---
# def meters_per_degree_lat_lon(lat_deg):
#     a = 6378137.0
#     f = 1.0 / 298.257223563
#     e2 = (2.0 - f) * f
#     lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
#     s = np.sin(lat); c = np.cos(lat)
#     one_minus_e2s2 = 1.0 - e2 * s*s
#     M = a * (1.0 - e2) / (one_minus_e2s2 ** 1.5)
#     N = a / np.sqrt(one_minus_e2s2)
#     m_per_deg_lat = (np.pi / 180.0) * M
#     m_per_deg_lon = (np.pi / 180.0) * N * np.clip(c, 0.0, None)
#     return m_per_deg_lat, m_per_deg_lon

# def step_lengths_for_rows(transform, H):
#     # velikost px ve stupních
#     deg_x = float(abs(transform.a))
#     deg_y = float(abs(transform.e))
#     lat0 = float(transform.f)
#     step_lat = float(transform.e)  # pro north-up záporné
#     lat_centers = lat0 + step_lat * (np.arange(H, dtype=np.float64) + 0.5)
#     mlat, mlon = meters_per_degree_lat_lon(lat_centers)
#     dx = mlon * deg_x
#     dy = mlat * deg_y
#     ddiag = np.hypot(dx, dy)
#     return dx.astype(np.float64), dy.astype(np.float64), ddiag.astype(np.float64)

# # --- D8 sousedé v pořadí: NE, E, SE, S, SW, W, NW, N ---
# D8 = [(-1, 1), (0, 1), (1, 1), (1, 0),
#       (1,-1), (0,-1), (-1,-1), (-1,0)]

# # --- SFD∞ (Tarboton 1997) => váhy do 8 sousedů (max 2 nenulové na buňku) ---
# def compute_flow_direction_sfd_inf(dem, transform, nodata_mask=None):
#     """
#     Vstup:
#         dem : 2D ndarray (float), hydrologicky opravený DEM (NaN = NoData)
#         transform : rasterio.Affine (north-up)
#         nodata_mask : 2D bool, True kde NoData (pokud None, odvodí se z ~isfinite)
#     Výstup:
#         flow : (H, W, 8) float32, váhy do D8 sousedů [NE,E,SE,S,SW,W,NW,N]
#     Pozn.: rozdělení mezi 2 buňky v plošce je lineární dle úhlu v plošce.
#     """
#     Z = np.asarray(dem, dtype=np.float64)
#     H, W = Z.shape
#     if nodata_mask is None:
#         nodata_mask = ~np.isfinite(Z)
#     else:
#         nodata_mask = np.asarray(nodata_mask, dtype=bool)

#     dx, dy, ddiag = step_lengths_for_rows(transform, H)
#     flow = np.zeros((H, W, 8), dtype=np.float32)

#     # map délky střed->soused pro daný řádek (index podle D8 pořadí)
#     def step_len_row(i):
#         return (ddiag[i],  dx[i], ddiag[i], dy[i],
#                 ddiag[i],  dx[i], ddiag[i], dy[i])

#     # plošky (okraje klínu): (n1, n2, typ_φ), kde φ = atan(dy/dx) nebo atan(dx/dy)
#     FACETS = [
#         (1, 0, 'c2d'),  # E - NE       φ = atan(dy/dx)
#         (0, 7, 'd2c'),  # NE - N       φ = atan(dx/dy)
#         (7, 6, 'c2d'),  # N - NW       φ = atan(dx/dy)  (cardinal first -> 'c2d', ale s dy/dx; viz níže)
#         (6, 5, 'd2c'),  # NW - W
#         (5, 4, 'c2d'),  # W - SW
#         (4, 3, 'd2c'),  # SW - S
#         (3, 2, 'c2d'),  # S - SE
#         (2, 1, 'd2c'),  # SE - E
#     ]
#     # Poznámka: pro (N-NW) a (S-SE) je φ = atan(dx/dy). Vyřešíme níže podmínkou.

#     for i in range(H):
#         sl = step_len_row(i)
#         # φ pro dvě varianty klínu v daném řádku
#         phi_c2d = np.arctan2(dy[i], dx[i])  # mezi cardinal->diagonal (E->NE, W->SW, ...): atan(dy/dx)
#         phi_d2c = np.arctan2(dx[i], dy[i])  # mezi diagonal->cardinal (NE->N, SW->S, ...): atan(dx/dy)

#         for j in range(W):
#             if nodata_mask[i, j]:
#                 continue

#             zc = Z[i, j]
#             best_s = 0.0
#             best_pair = None  # (k1, k2, r, φ)

#             # Projeď 8 plošek (klínů)
#             for (k1, k2, kind) in FACETS:
#                 di1, dj1 = D8[k1]
#                 di2, dj2 = D8[k2]
#                 n1i, n1j = i + di1, j + dj1
#                 n2i, n2j = i + di2, j + dj2

#                 ok1 = (0 <= n1i < H) and (0 <= n1j < W) and (not nodata_mask[n1i, n1j])
#                 ok2 = (0 <= n2i < H) and (0 <= n2j < W) and (not nodata_mask[n2i, n2j])

#                 # pokud oba chybí, plošku přeskoč
#                 if not ok1 and not ok2:
#                     continue

#                 # φ pro tento klín
#                 phi = phi_c2d if kind == 'c2d' else phi_d2c
#                 if phi <= 0.0:
#                     continue  # degenerace, nemělo by nastat

#                 # sklony podél hran klínu (kladné = dolů z centra)
#                 s1 = (zc - Z[n1i, n1j]) / sl[k1] if ok1 else -np.inf
#                 s2 = (zc - Z[n2i, n2j]) / sl[k2] if ok2 else -np.inf

#                 # pokud oba sklony <= 0, v tomto klínu žádný spád
#                 if (s1 <= 0.0) and (s2 <= 0.0):
#                     continue

#                 # úhel v klínu r = atan2(s2, s1), omez na [0, φ]
#                 r = np.arctan2(max(s2, 0.0), max(s1, 0.0))  # záporné nahradíme 0 kvůli směru v klínu
#                 if r < 0.0:
#                     r = 0.0
#                 if r > phi:
#                     r = phi

#                 # velikost spádu v klínu:
#                 #   uvnitř: sqrt(s1^2 + s2^2)
#                 #   na hranách: odpovídající s1 nebo s2
#                 if r == 0.0:
#                     s = max(s1, 0.0)
#                 elif r == phi:
#                     s = max(s2, 0.0)
#                 else:
#                     # obě složky kladné (po ořezu); pokud by některá byla -inf, už bychom byli na hraně
#                     s = np.hypot(max(s1, 0.0), max(s2, 0.0))

#                 if s > best_s:
#                     best_s = s
#                     best_pair = (k1, k2, r, phi, ok1, ok2)

#             # výstupní váhy – maximálně dva směry z best_pair
#             if best_pair is None or best_s <= 0.0:
#                 continue  # žádný odtok z buňky

#             k1, k2, r, phi, ok1, ok2 = best_pair
#             if r <= 0.0 or not ok2:
#                 flow[i, j, k1] = 1.0  # vše do hrany 1
#             elif r >= phi or not ok1:
#                 flow[i, j, k2] = 1.0  # vše do hrany 2
#             else:
#                 w2 = float(r / phi)
#                 w1 = 1.0 - w2
#                 flow[i, j, k1] = np.float32(w1)
#                 flow[i, j, k2] = np.float32(w2)

#     # NoData buňky – nula vah
#     flow[nodata_mask, :] = 0.0
#     return flow

from __future__ import annotations

import numpy as np

# D8 neighbor offsets in order: [NE, E, SE, S, SW, W, NW, N]
D8_OFFSETS = [
    (-1,  1), (0,  1), (1,  1), (1,  0),
    ( 1, -1), (0, -1), (-1, -1), (-1,  0),
]

# Cardinal directions indices in this D8 ordering: E=1, S=3, W=5, N=7
CARDINAL_IDXS = {1, 3, 5, 7}


def meters_per_degree_lat_lon(lat_deg: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Meters per 1 degree of latitude/longitude at given latitude(s), WGS84."""
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = (2.0 - f) * f

    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    s = np.sin(lat)
    c = np.cos(lat)

    one_minus_e2s2 = 1.0 - e2 * s * s
    M = a * (1.0 - e2) / (one_minus_e2s2 ** 1.5)
    N = a / np.sqrt(one_minus_e2s2)

    m_per_deg_lat = (np.pi / 180.0) * M
    m_per_deg_lon = (np.pi / 180.0) * N * np.clip(c, 0.0, None)
    return m_per_deg_lat, m_per_deg_lon


def step_lengths_for_rows_epsg4326(transform, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-row step lengths between pixel centers in meters for a north-up EPSG:4326 grid.
    """
    deg_x = float(abs(transform.a))
    deg_y = float(abs(transform.e))

    lat_origin = float(transform.f)
    lat_step = float(transform.e)  # negative for north-up
    lat_centers = lat_origin + lat_step * (np.arange(height, dtype=np.float64) + 0.5)

    mlat, mlon = meters_per_degree_lat_lon(lat_centers)
    dx = mlon * deg_x
    dy = mlat * deg_y
    d_diag = np.hypot(dx, dy)
    return dx, dy, d_diag


def compute_flow_direction_sfd_infinity(
    dem: np.ndarray,
    transform,
    *,
    nodata_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    D-infinity / SFD-infinity flow direction (Tarboton-style): at most 2 non-zero outflow
    weights per cell to D8 neighbors.

    Returns
    -------
    flow_weights : (H, W, 8) float32
        Weights for directions [NE, E, SE, S, SW, W, NW, N].
    """
    z = np.asarray(dem, dtype=np.float64)
    height, width = z.shape

    if nodata_mask is None:
        nodata_mask = ~np.isfinite(z)
    else:
        nodata_mask = np.asarray(nodata_mask, dtype=bool)

    dx_row, dy_row, d_diag_row = step_lengths_for_rows_epsg4326(transform, height)

    flow = np.zeros((height, width, 8), dtype=np.float32)

    # Neighbor-to-neighbor facets around the cell (adjacent directions in this D8 ordering)
    # (k1 -> k2) goes around clockwise in index order.
    facets = [(k, (k + 1) % 8) for k in range(8)]

    for i in range(height):
        # Step lengths in D8 order for this row (meters)
        step_len = np.array(
            [d_diag_row[i], dx_row[i], d_diag_row[i], dy_row[i],
             d_diag_row[i], dx_row[i], d_diag_row[i], dy_row[i]],
            dtype=np.float64,
        )

        for j in range(width):
            if nodata_mask[i, j]:
                continue

            zc = z[i, j]

            best_s = 0.0
            best = None  # (k1, k2, r, phi)

            for k1, k2 in facets:
                di1, dj1 = D8_OFFSETS[k1]
                di2, dj2 = D8_OFFSETS[k2]
                n1i, n1j = i + di1, j + dj1
                n2i, n2j = i + di2, j + dj2

                ok1 = (0 <= n1i < height) and (0 <= n1j < width) and (not nodata_mask[n1i, n1j])
                ok2 = (0 <= n2i < height) and (0 <= n2j < width) and (not nodata_mask[n2i, n2j])
                if not ok1 and not ok2:
                    continue

                # Slopes along the two facet edges (positive = downslope)
                s1 = (zc - z[n1i, n1j]) / step_len[k1] if ok1 else -np.inf
                s2 = (zc - z[n2i, n2j]) / step_len[k2] if ok2 else -np.inf
                if (s1 <= 0.0) and (s2 <= 0.0):
                    continue

                # Determine facet angle phi:
                # - If one direction is cardinal and the other diagonal -> phi = atan(dy/dx) or atan(dx/dy)
                # In EPSG:4326, dx != dy, so choose based on whether the cardinal is E/W or N/S.
                k1_is_card = k1 in CARDINAL_IDXS
                k2_is_card = k2 in CARDINAL_IDXS

                if k1_is_card ^ k2_is_card:
                    # One cardinal, one diagonal
                    # If the cardinal is E/W => use atan(dy/dx), if N/S => atan(dx/dy)
                    cardinal_k = k1 if k1_is_card else k2
                    if cardinal_k in (1, 5):  # E or W
                        phi = float(np.arctan2(dy_row[i], dx_row[i]))
                    else:  # N or S
                        phi = float(np.arctan2(dx_row[i], dy_row[i]))
                else:
                    # This should not happen with correct adjacent D8 facets, but keep safe:
                    # treat as a 45-degree wedge in metric sense
                    phi = float(np.pi / 4.0)

                if phi <= 0.0:
                    continue

                # Direction within the facet (clamp to [0, phi])
                r = float(np.arctan2(max(s2, 0.0), max(s1, 0.0)))
                r = min(max(r, 0.0), phi)

                # Facet slope magnitude
                if r == 0.0:
                    s = max(s1, 0.0)
                elif r == phi:
                    s = max(s2, 0.0)
                else:
                    s = float(np.hypot(max(s1, 0.0), max(s2, 0.0)))

                if s > best_s:
                    best_s = s
                    best = (k1, k2, r, phi, ok1, ok2)

            if best is None or best_s <= 0.0:
                continue

            k1, k2, r, phi, ok1, ok2 = best

            # Convert r to two neighbor weights (linear split within the facet)
            if (r <= 0.0) or (not ok2):
                flow[i, j, k1] = 1.0
            elif (r >= phi) or (not ok1):
                flow[i, j, k2] = 1.0
            else:
                w2 = r / phi
                flow[i, j, k1] = np.float32(1.0 - w2)
                flow[i, j, k2] = np.float32(w2)

    flow[nodata_mask, :] = 0.0
    return flow

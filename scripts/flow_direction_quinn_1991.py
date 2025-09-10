import numpy as np

# --- Helpers: meters per degree & per-row step lengths (EPSG:4326) ---
def meters_per_degree_lat_lon(lat_deg):
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = (2.0 - f) * f
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    s = np.sin(lat); c = np.cos(lat)
    one_minus_e2s2 = 1.0 - e2 * s*s
    M = a * (1.0 - e2) / (one_minus_e2s2 ** 1.5)
    N = a / np.sqrt(one_minus_e2s2)
    m_per_deg_lat = (np.pi / 180.0) * M
    m_per_deg_lon = (np.pi / 180.0) * N * np.clip(c, 0.0, None)
    return m_per_deg_lat, m_per_deg_lon

def step_lengths_for_rows(transform, H):
    # pixel sizes in degrees (north-up assumed: a>0, e<0)
    deg_x = float(abs(transform.a))
    deg_y = float(abs(transform.e))
    lat0 = float(transform.f)
    step_lat = float(transform.e)  # negative for north-up
    lat_centers_deg = lat0 + step_lat * (np.arange(H, dtype=np.float64) + 0.5)
    mlat, mlon = meters_per_degree_lat_lon(lat_centers_deg)
    dx = mlon * deg_x
    dy = mlat * deg_y
    ddiag = np.hypot(dx, dy)
    return dx.astype(np.float64), dy.astype(np.float64), ddiag.astype(np.float64)

# D8 sousedé v pořadí: NE, E, SE, S, SW, W, NW, N
D8 = [(-1, 1), (0, 1), (1, 1), (1, 0),
      (1,-1), (0,-1), (-1,-1), (-1, 0)]

# Quinn 1991 MFD (FD8): w_i ∝ L_i * (tanβ_i)^p  ; default p=1 (čistý Quinn 1991)
def compute_flow_direction_quinn_1991(dem, transform, *, p=1.0, nodata_mask=None):
    """
    Vstup:
        dem : 2D float (NaN = NoData), hydro-korigovaný DEM
        transform : rasterio.Affine (north-up)
        p : exponent; p=1 odpovídá Quinn 1991 (Holmgren je obecnější p>1)
        nodata_mask : 2D bool (pokud None, odvozeno z ~isfinite)
    Výstup:
        flow : (H, W, 8) float32, váhy do [NE,E,SE,S,SW,W,NW,N]
    """
    Z = np.asarray(dem, dtype=np.float64)
    H, W = Z.shape
    if nodata_mask is None:
        nodata_mask = ~np.isfinite(Z)
    else:
        nodata_mask = np.asarray(nodata_mask, dtype=bool)

    # metrické vzdálenosti mezi středy buněk (po řádcích)
    dx, dy, ddiag = step_lengths_for_rows(transform, H)

    # efektivní délky vrstevnice (konstanty podle Quinn 1991 / FD8):
    L_card = 0.5
    L_diag = np.sqrt(2.0) / 4.0
    # map L_i do našeho D8 pořadí: NE(diag),E(card),SE(diag),S(card),SW(diag),W(card),NW(diag),N(card)
    L_vec = np.array([L_diag, L_card, L_diag, L_card, L_diag, L_card, L_diag, L_card], dtype=np.float64)

    flow = np.zeros((H, W, 8), dtype=np.float32)

    for i in range(H):
        # délky kroků k sousedům v tomto řádku (souřadnice NE..N)
        step_len = (ddiag[i],  dx[i], ddiag[i], dy[i],
                    ddiag[i],  dx[i], ddiag[i], dy[i])

        for j in range(W):
            if nodata_mask[i, j]:
                continue

            zc = Z[i, j]
            w_list = []
            idxs = []

            for k, (di, dj) in enumerate(D8):
                ni, nj = i + di, j + dj
                if not (0 <= ni < H and 0 <= nj < W): 
                    continue
                if nodata_mask[ni, nj]:
                    continue

                d = step_len[k]
                if d <= 0:
                    continue

                dz = zc - Z[ni, nj]
                if dz <= 0.0:
                    continue  # jen kladné spády

                tanb = dz / d
                w = L_vec[k] * (tanb ** float(p))
                if w > 0.0:
                    w_list.append(w)
                    idxs.append(k)

            if not idxs:
                continue

            s = float(np.sum(w_list))
            if s > 0.0:
                inds = np.array(idxs, dtype=int)
                flow[i, j, inds] = (np.array(w_list, dtype=np.float64) / s).astype(np.float32)

    flow[nodata_mask, :] = 0.0
    return flow

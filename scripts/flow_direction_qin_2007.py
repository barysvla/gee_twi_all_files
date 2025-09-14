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

# D8 neighbors ordered: [NE, E, SE, S, SW, W, NW, N]
D8 = [(-1, 1), (0, 1), (1, 1), (1, 0),
      (1,-1), (0,-1), (-1,-1), (-1, 0)]

def compute_flow_direction_qin_2007(
    dem,
    transform,
    *,
    nodata_mask=None,
    pl: float = 1.1,          # lower bound of exponent
    pu: float = 10.0,         # upper bound of exponent
    emax: float = 1.0,        # saturation for e = max(tanβ); default tan(45°)=1
    renormalize: bool = True
):
    """
    Qin et al. (2007) MFD-md flow direction (weights) using adaptive exponent p(e).
    - Weights: w_i ∝ L_i * (tanβ_i)^{p(e)}, for all downslope neighbors.
    - L_i are contour-length factors: L_card = 0.5, L_diag = sqrt(2)/4 (Quinn/Freeman convention).
    - e = max(tanβ_i) over downslope neighbors; p(e) linearly maps [0, emax] -> [pl, pu].

    Inputs
    ------
    dem : 2D float array (NaN = NoData), hydrologically conditioned.
    transform : rasterio.Affine (north-up, degrees). Metric step lengths are computed per row.
    nodata_mask : 2D bool | None. If None, derived from ~isfinite(dem).
    pl, pu : float. Lower/upper bounds for adaptive exponent.
    emax : float. Slope saturation in tan units (default 1 = tan 45°).
    renormalize : bool. If True, per-cell positive weights are normalized to 1.

    Output
    ------
    flow : (H, W, 8) float32 weights to [NE,E,SE,S,SW,W,NW,N].
    """
    Z = np.asarray(dem, dtype=np.float64)
    H, W = Z.shape
    if nodata_mask is None:
        nodata_mask = ~np.isfinite(Z)
    else:
        nodata_mask = np.asarray(nodata_mask, dtype=bool)

    # Metric step lengths per row from degrees (EPSG:4326)
    dx, dy, ddiag = step_lengths_for_rows(transform, H)

    # Contour-length weights (Quinn/Freeman)
    L_card = 0.5
    L_diag = np.sqrt(2.0) / 4.0
    L_vec = np.array([L_diag, L_card, L_diag, L_card, L_diag, L_card, L_diag, L_card],
                     dtype=np.float64)

    flow = np.zeros((H, W, 8), dtype=np.float32)

    for i in range(H):
        # Step length to neighbors in this row (order NE..N)
        step_len = (ddiag[i],  dx[i], ddiag[i], dy[i],
                    ddiag[i],  dx[i], ddiag[i], dy[i])

        for j in range(W):
            if nodata_mask[i, j]:
                continue

            zc = Z[i, j]

            # Compute tanβ_i for all downslope neighbors
            tanb = np.zeros(8, dtype=np.float64)
            has_pos = False
            for k, (di, dj) in enumerate(D8):
                ni, nj = i + di, j + dj
                if not (0 <= ni < H and 0 <= nj < W) or nodata_mask[ni, nj]:
                    continue
                d = step_len[k]
                if d <= 0:
                    continue
                dz = zc - Z[ni, nj]
                if dz > 0.0:               # downslope only
                    tb = dz / d            # tan(beta)
                    if tb > 0.0 and np.isfinite(tb):
                        tanb[k] = tb
                        has_pos = True

            if not has_pos:
                # no downslope neighbor -> no outflow
                continue

            # Adaptive exponent p(e) with e = max(tanβ)
            e = float(np.max(tanb))
            e_clamped = min(e / float(emax), 1.0) if emax > 0 else 1.0
            p = float(pl + (pu - pl) * e_clamped)

            # Raw weights w_i ∝ L_i * (tanβ_i)^p for positive tanβ_i
            with np.errstate(over='ignore', invalid='ignore'):
                w = L_vec * np.power(tanb, p, dtype=np.float64)
            w[~np.isfinite(w)] = 0.0
            w[w < 0.0] = 0.0

            s = w.sum(dtype=np.float64)
            if s > 0.0:
                flow[i, j, :] = (w / s).astype(np.float32)

    # Ensure NoData cells have zero outflow
    flow[nodata_mask, :] = 0.0

    if renormalize:
        sums = flow.sum(axis=2, dtype=np.float32)
        pos = (sums > 0.0) & (~nodata_mask)
        flow[pos, :] /= sums[pos, None]

    return flow

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

# D8 neighbors ordered as: [NE, E, SE, S, SW, W, NW, N]
D8 = [(-1, 1), (0, 1), (1, 1), (1, 0),
      (1,-1), (0,-1), (-1,-1), (-1, 0)]

# Triangular facets between D8 neighbors (k1, k2) with the angle type:
# 'c2d' => angle φ = atan(dy/dx); 'd2c' => angle φ = atan(dx/dy)
FACETS = [
    (1, 0, 'c2d'),  # E - NE
    (0, 7, 'd2c'),  # NE - N
    (7, 6, 'c2d'),  # N - NW
    (6, 5, 'd2c'),  # NW - W
    (5, 4, 'c2d'),  # W - SW
    (4, 3, 'd2c'),  # SW - S
    (3, 2, 'c2d'),  # S - SE
    (2, 1, 'd2c'),  # SE - E
]

def compute_flow_direction_md_infinity(
    dem,
    transform,
    *,
    nodata_mask=None,
    clamp_eps=1e-12,
    renormalize=True
):
    """
    MD-Infinity (Seibert & McGlynn, 2007) triangular multi-flow direction.

    Inputs
    ------
    dem : 2D float array
        Hydrologically conditioned DEM (NaN = NoData).
    transform : rasterio.Affine
        North-up geotransform in EPSG:4326 (degrees). Metric step lengths are computed per row.
    nodata_mask : 2D bool or None
        True where NoData. If None, derived as ~np.isfinite(dem).
    clamp_eps : float
        Numerical clamp to keep r in [0, φ] and avoid tiny negative/overshoot due to FP error.
    renormalize : bool
        If True, re-normalize positive outflow weights of each cell to sum to 1.

    Output
    ------
    flow : (H, W, 8) float32
        Outflow weights to D8 neighbors [NE,E,SE,S,SW,W,NW,N]. Sum per cell ∈ {0,1}.
        Cells with no positive downslope facet keep all zeros.

    Method
    ------
    For each cell:
      1) Evaluate all 8 triangular facets (as in D∞) and compute their positive slope magnitude s_f.
      2) Keep facets with s_f > 0. Let S = Σ_f s_f.
      3) For each such facet, assign a facet-share W_f = s_f / S.
      4) Split W_f between the two facet neighbors linearly by r/φ (D∞ rule).
      5) Sum contributions over facets -> up to 8 non-zero neighbors.
    """
    Z = np.asarray(dem, dtype=np.float64)
    H, W = Z.shape
    if nodata_mask is None:
        nodata_mask = ~np.isfinite(Z)
    else:
        nodata_mask = np.asarray(nodata_mask, dtype=bool)

    # Per-row metric step lengths from degrees (EPSG:4326)
    dx, dy, ddiag = step_lengths_for_rows(transform, H)

    flow = np.zeros((H, W, 8), dtype=np.float32)

    for i in range(H):
        # Step lengths to D8 neighbors for this row (order NE..N)
        step_len = (ddiag[i],  dx[i], ddiag[i], dy[i],
                    ddiag[i],  dx[i], ddiag[i], dy[i])

        # Facet angles for this row
        phi_c2d = np.arctan2(dy[i], dx[i])  # cardinal->diagonal
        phi_d2c = np.arctan2(dx[i], dy[i])  # diagonal->cardinal

        for j in range(W):
            if nodata_mask[i, j]:
                continue

            zc = Z[i, j]

            # Collect downslope facets: (k1, k2, φ, r, s, ok1, ok2)
            facets = []
            for (k1, k2, kind) in FACETS:
                di1, dj1 = D8[k1]
                di2, dj2 = D8[k2]
                n1i, n1j = i + di1, j + dj1
                n2i, n2j = i + di2, j + dj2

                ok1 = (0 <= n1i < H) and (0 <= n1j < W) and (not nodata_mask[n1i, n1j])
                ok2 = (0 <= n2i < H) and (0 <= n2j < W) and (not nodata_mask[n2i, n2j])

                # Skip facet if both neighbors are invalid/OOB
                if not ok1 and not ok2:
                    continue

                phi = phi_c2d if kind == 'c2d' else phi_d2c
                if not np.isfinite(phi) or phi <= 0.0:
                    continue  # degenerate metric geometry

                # Edge-aligned slopes from center to each neighbor along facet edges
                s1 = (zc - Z[n1i, n1j]) / step_len[k1] if ok1 else -np.inf
                s2 = (zc - Z[n2i, n2j]) / step_len[k2] if ok2 else -np.inf

                # If both edges are non-descending, facet does not drain
                if (s1 <= 0.0) and (s2 <= 0.0):
                    continue

                # Angular position r inside the facet wedge, clamp to [0, φ]
                r = np.arctan2(max(s2, 0.0), max(s1, 0.0))
                if r < 0.0: r = 0.0
                if r > phi: r = phi

                # Facet slope magnitude s_f:
                #  - on edges => use respective positive edge slope
                #  - inside   => Euclidean norm of positive components
                if r <= clamp_eps:
                    s = max(s1, 0.0)
                elif (phi - r) <= clamp_eps:
                    s = max(s2, 0.0)
                else:
                    s = np.hypot(max(s1, 0.0), max(s2, 0.0))

                if s > 0.0 and np.isfinite(s):
                    facets.append((k1, k2, phi, r, s, ok1, ok2))

            if not facets:
                # no positive downslope facet -> no outflow from this cell
                continue

            # Distribute proportionally to facet slopes
            S = float(sum(f[4] for f in facets))
            if S <= 0.0 or not np.isfinite(S):
                continue

            for (k1, k2, phi, r, s, ok1, ok2) in facets:
                Wf = s / S  # facet share

                # Split Wf within facet between its two neighbors (D∞ linear rule)
                if (not ok2) or (r <= clamp_eps):
                    flow[i, j, k1] += np.float32(Wf)
                elif (not ok1) or ((phi - r) <= clamp_eps):
                    flow[i, j, k2] += np.float32(Wf)
                else:
                    w2 = float(r / phi)
                    w1 = 1.0 - w2
                    flow[i, j, k1] += np.float32(Wf * w1)
                    flow[i, j, k2] += np.float32(Wf * w2)

    # Zero weights on NoData cells
    flow[nodata_mask, :] = 0.0

    if renormalize:
        # Re-normalize positive weights per cell to sum to 1 (guards FP drift)
        sums = flow.sum(axis=2, dtype=np.float32)
        pos = (sums > 0.0) & (~nodata_mask)
        flow[pos, :] /= sums[pos, None]

    return flow

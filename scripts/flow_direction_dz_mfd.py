import numpy as np

# D8 neighbors in your order: NE, E, SE, S, SW, W, NW, N
D8 = [(-1, 1), (0, 1), (1, 1), (1, 0),
      (1,-1), (0,-1), (-1,-1), (-1, 0)]

def compute_flow_direction_dz_mfd(dem, *, p=1.6, nodata_mask=None):
    """
    MFD s vážením podle výškového poklesu: w_k ∝ (Δz_k)^p, Δz_k = z_center - z_neighbor.
    - Bez CIT, bez L_i, bez metrických vzdáleností (Δz už je 'direction-free').
    - Rychlé, jednorázové, dává konvergentnější (SFD-bližší) směry.

    Parametry:
        dem : 2D float (NaN = NoData), hydrologicky opravený DEM
        p   : exponent (větší p → více konvergentní, blíže D8)
        nodata_mask : 2D bool (pokud None, odvozeno z ~isfinite)

    Návrat:
        flow : (H, W, 8) float32, váhy do sousedů [NE,E,SE,S,SW,W,NW,N]
    """
    Z = np.asarray(dem, dtype=np.float64)
    H, W = Z.shape

    if nodata_mask is None:
        nodata_mask = ~np.isfinite(Z)
    else:
        nodata_mask = np.asarray(nodata_mask, bool)

    flow = np.zeros((H, W, 8), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            if nodata_mask[i, j]:
                continue

            zc = Z[i, j]
            idxs = []
            w = []

            # positive drops only
            for k, (di, dj) in enumerate(D8):
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and not nodata_mask[ni, nj]:
                    dz = zc - Z[ni, nj]
                    if dz > 0.0:
                        idxs.append(k)
                        w.append(dz ** float(p))

            if not idxs:
                continue

            s = float(np.sum(w))
            if s > 0.0:
                inds = np.array(idxs, dtype=int)
                flow[i, j, inds] = (np.array(w, dtype=np.float64) / s).astype(np.float32)

    flow[nodata_mask, :] = 0.0
    return flow

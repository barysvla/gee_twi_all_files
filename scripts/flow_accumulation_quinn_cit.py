import numpy as np
from collections import deque

def compute_flow_accumulation_quinn_cit(flow_quinn_cit,
                                        pixel_area_m2=None,
                                        nodata_mask=None,
                                        normalize_weights=True,
                                        out='km2'):
    """
    Flow accumulation (Quinn 1995, CIT weights).

    Parameters:
        flow_quinn_cit : (H, W, 8) float32
            Flow fractions to 8 neighbors in D8 order (sum<=1 per cell).
        pixel_area_m2  : (H, W) float32 or float or None
            Per-pixel area in m^2 (array or scalar). Required for 'm2'/'km2'.
            Ignored for 'cells'.
        nodata_mask    : (H, W) bool or None
            True where cell is invalid (NoData). If None, assumed all valid.
        normalize_weights : bool
            If True, renormalize positive weights per-cell to sum to 1.
        out            : {'cells','m2','km2'}
            Output units.

    Returns:
        (H, W) float32 accumulation in requested units.
    """
    H, W, K = flow_quinn_cit.shape
    assert K == 8, "Expected 8 neighbor weights (D8)."

    if nodata_mask is None:
        nodata_mask = np.zeros((H, W), dtype=bool)
    else:
        nodata_mask = nodata_mask.astype(bool, copy=False)

    # Copy weights and zero out invalid sources
    Wgt = flow_quinn_cit.astype(np.float32, copy=True)
    Wgt[nodata_mask, :] = 0.0

    # Optional per-cell normalization of outgoing weights
    if normalize_weights:
        sums = Wgt.sum(axis=2, dtype=np.float32)
        pos = (sums > 0) & (~nodata_mask)
        # vyhnout se dělení nulou
        Wgt[pos, :] /= sums[pos, None]

    # D8 neighbor offsets (bez int8!)
    D8 = [(-1, 1), (0, 1), (1, 1), (1, 0),
          (1, -1), (0, -1), (-1, -1), (-1, 0)]

    # Inicializace akumulace
    if out == 'cells':
        acc = np.ones((H, W), dtype=np.float64)
    elif out in ('m2', 'km2'):
        if pixel_area_m2 is None:
            raise ValueError("pixel_area_m2 is required for out='m2' or 'km2'.")
        if np.isscalar(pixel_area_m2):
            acc = np.full((H, W), float(pixel_area_m2), dtype=np.float64)
        else:
            assert pixel_area_m2.shape == (H, W), "pixel_area_m2 must match grid shape"
            acc = pixel_area_m2.astype(np.float64, copy=True)
    else:
        raise ValueError("out must be one of {'cells','m2','km2'}")

    # Invalidate NoData cells
    acc[nodata_mask] = 0.0

    # In-degree (počet přítoků) pro topologické zpracování
    indeg = np.zeros((H, W), dtype=np.int32)
    for i in range(H):
        for j in range(W):
            if nodata_mask[i, j]:
                continue
            for k, (di, dj) in enumerate(D8):
                w = Wgt[i, j, k]
                if w <= 0.0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and not nodata_mask[ni, nj]:
                    indeg[ni, nj] += 1

    # Kahn-like topological processing
    q = deque((i, j) for i in range(H) for j in range(W)
              if (indeg[i, j] == 0 and not nodata_mask[i, j]))

    while q:
        i, j = q.popleft()
        a_ij = acc[i, j]
        if a_ij == 0.0:
            # nic neteče dál
            pass
        for k, (di, dj) in enumerate(D8):
            w = Wgt[i, j, k]
            if w <= 0.0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and not nodata_mask[ni, nj]:
                acc[ni, nj] += a_ij * w
                indeg[ni, nj] -= 1
                if indeg[ni, nj] == 0:
                    q.append((ni, nj))

    if out == 'km2':
        acc /= 1e6  # m^2 -> km^2

    return acc.astype(np.float32)

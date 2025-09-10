import numpy as np
from collections import deque

# D8 neighbor offsets in order [NE, E, SE, S, SW, W, NW, N]
D8 = [(-1, 1), (0, 1), (1, 1), (1, 0),
      (1,-1), (0,-1), (-1,-1), (-1, 0)]

def compute_flow_accumulation_sfd_inf(flow_weights,
                            pixel_area_m2=None,
                            nodata_mask=None,
                            normalize=True,
                            out='cells',
                            cycle_check=True):
    """
    Generic accumulation over directional weights (works for SFD∞ and MFD).
    Parameters
    ----------
    flow_weights : (H, W, 8) float
        Non-negative outflow weights to D8 neighbors; per-cell sum can be <= 1.
    pixel_area_m2 : None | float | (H, W) float
        Per-pixel area in m^2 (scalar or array). Required for 'm2'/'km2'. Ignored for 'cells'.
    nodata_mask : (H, W) bool or None
        True where the cell is invalid (NoData). If None, all cells are considered valid.
    normalize : bool
        If True, renormalize positive per-cell weights to sum to 1 (recommended).
    out : {'cells','m2','km2'}
        Output units.
    cycle_check : bool
        If True, check for unresolved cycles (should not occur if flats were resolved).

    Returns
    -------
    acc : (H, W) float32
        Flow accumulation in requested units.
    """
    H, W, K = flow_weights.shape
    if K != 8:
        raise ValueError("flow_weights must have shape (H, W, 8)")

    if nodata_mask is None:
        nodata_mask = np.zeros((H, W), dtype=bool)
    else:
        nodata_mask = np.asarray(nodata_mask, bool)

    # 1) Sanitize weights
    Wgt = np.nan_to_num(flow_weights.astype(np.float32, copy=True),
                        nan=0.0, posinf=0.0, neginf=0.0)
    Wgt = np.maximum(Wgt, 0.0)
    Wgt[nodata_mask, :] = 0.0

    # 2) Optional per-cell renormalization
    if normalize:
        sums = Wgt.sum(axis=2, dtype=np.float32)
        pos = (sums > 0) & (~nodata_mask)
        Wgt[pos, :] /= sums[pos, None]

    # 3) Initialize accumulation
    if out == 'cells':
        acc = np.ones((H, W), dtype=np.float64)
    elif out in ('m2', 'km2'):
        if pixel_area_m2 is None:
            raise ValueError("pixel_area_m2 is required for out='m2' or 'km2'.")
        acc = (np.full((H, W), float(pixel_area_m2), dtype=np.float64)
               if np.isscalar(pixel_area_m2)
               else np.asarray(pixel_area_m2, dtype=np.float64).copy())
    else:
        raise ValueError("out must be 'cells', 'm2', or 'km2'")
    acc[nodata_mask] = 0.0

    # 4) In-degree for topological processing (number of incoming edges)
    indeg = np.zeros((H, W), dtype=np.int32)
    for i in range(H):
        for j in range(W):
            if nodata_mask[i, j]:
                continue
            for k, (di, dj) in enumerate(D8):
                if Wgt[i, j, k] <= 0.0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and not nodata_mask[ni, nj]:
                    indeg[ni, nj] += 1

    # 5) Kahn queue (topological order)
    q = deque((i, j) for i in range(H) for j in range(W)
              if (indeg[i, j] == 0 and not nodata_mask[i, j]))
    visited = 0

    while q:
        i, j = q.popleft()
        visited += 1
        a = acc[i, j]
        if a != 0.0:
            for k, (di, dj) in enumerate(D8):
                w = Wgt[i, j, k]
                if w <= 0.0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and not nodata_mask[ni, nj]:
                    acc[ni, nj] += a * w
                    indeg[ni, nj] -= 1
                    if indeg[ni, nj] == 0:
                        q.append((ni, nj))

    # 6) Optional cycle check (indicates unresolved flats/sinks)
    if cycle_check:
        total_valid = int((~nodata_mask).sum())
        if visited != total_valid:
            raise RuntimeError("Cycle detected (unresolved flats?). Resolve flats (epsilon) before accumulation.")

    if out == 'km2':
        acc *= 1e-6  # m² -> km²

    return acc.astype(np.float32)

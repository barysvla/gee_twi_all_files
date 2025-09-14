import numpy as np
from collections import deque

# D8 neighbors in the order [NE, E, SE, S, SW, W, NW, N]
D8 = [(-1, 1), (0, 1), (1, 1), (1, 0),
      (1,-1), (0,-1), (-1,-1), (-1, 0)]

def compute_flow_accumulation_md_infinity(
    flow_weights,
    *,
    pixel_area_m2=None,        # scalar or (H,W); required for 'm2'/'km2'/'sca'
    nodata_mask=None,
    normalize: bool = True,    # per-cell renormalization of positive weights to sum to 1
    out: str = 'cells',        # 'cells' | 'm2' | 'km2' | 'sca'
    cycle_check: bool = True,
    dx_per_row=None            # (H,) meters; required for 'sca' (pass from step_lengths_for_rows)
):
    """
    Topological flow accumulation for MD-Infinity (or any MFD) given precomputed outflow weights.

    Parameters
    ----------
    flow_weights : (H, W, 8) float
        Outflow proportions to D8 neighbors [NE,E,SE,S,SW,W,NW,N]. Can be sparse per cell.
    pixel_area_m2 : None | float | (H, W) float
        Per-pixel area in m^2. Needed for 'm2', 'km2', and 'sca'.
    nodata_mask : (H, W) bool or None
        True where NoData; these cells neither contribute nor receive flow.
    normalize : bool
        If True, re-normalize positive per-cell weights to sum to 1 (guards FP drift).
    out : str
        'cells' = unit contributions, 'm2' = area, 'km2' = area in km^2,
        'sca' = Specific Catchment Area = area [m^2] / contour length [m].
    cycle_check : bool
        If True, raises if a cycle is detected (unresolved flats/sinks).
    dx_per_row : (H,) float
        Contour length per pixel in meters for each row (e.g., from step_lengths_for_rows).
        Required for 'sca'.

    Returns
    -------
    acc : (H, W) float32
        Accumulation in requested units.

    Notes
    -----
    - Works with MD-Infinity weights produced by a triangular facet router.
    - Assumes flats/sinks have been resolved upstream and weights are non-negative.
    """
    Wgt = np.nan_to_num(np.asarray(flow_weights, dtype=np.float32),
                        nan=0.0, posinf=0.0, neginf=0.0)
    if Wgt.ndim != 3 or Wgt.shape[2] != 8:
        raise ValueError("flow_weights must have shape (H, W, 8)")
    H, W, _ = Wgt.shape

    if nodata_mask is None:
        nodata_mask = np.zeros((H, W), dtype=bool)
    else:
        nodata_mask = np.asarray(nodata_mask, dtype=bool)

    # Clamp to [0, +inf) and mask NoData
    Wgt = np.maximum(Wgt, 0.0)
    Wgt[nodata_mask, :] = 0.0

    # Optional per-cell renormalization
    if normalize:
        sums = Wgt.sum(axis=2, dtype=np.float32)
        pos = (sums > 0.0) & (~nodata_mask)
        Wgt[pos, :] /= sums[pos, None]

    # Initialize accumulation (float64 for stability)
    if out in ('cells', 'sca'):
        acc = np.ones((H, W), dtype=np.float64)
    elif out in ('m2', 'km2'):
        if pixel_area_m2 is None:
            raise ValueError("pixel_area_m2 is required for out='m2' or 'km2'.")
        acc = (np.full((H, W), float(pixel_area_m2), dtype=np.float64)
               if np.isscalar(pixel_area_m2)
               else np.asarray(pixel_area_m2, dtype=np.float64).copy())
    else:
        raise ValueError("out must be 'cells', 'm2', 'km2', or 'sca'")
    acc[nodata_mask] = 0.0

    # In-degree for Kahn topological ordering
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

    # Kahn queue (zero in-degree first)
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

    if cycle_check:
        total_valid = int((~nodata_mask).sum())
        if visited != total_valid:
            raise RuntimeError("Cycle detected (unresolved flats/sinks). Resolve flats before accumulation.")

    # Output units
    if out == 'km2':
        acc *= 1e-6
    elif out == 'sca':
        if pixel_area_m2 is None:
            raise ValueError("pixel_area_m2 is required for out='sca'.")
        if dx_per_row is None:
            raise ValueError("dx_per_row (meters per row) is required for out='sca'.")
        dx = np.asarray(dx_per_row, dtype=np.float64)
        if dx.shape != (H,):
            raise ValueError("dx_per_row must have shape (H,).")
        acc = acc / dx[:, None]  # SCA = contributing area [m^2] / contour length [m]

    return acc.astype(np.float32)

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

# Sanitaze
# def compute_flow_accumulation_quinn_cit(flow_quinn_cit,
#                                         pixel_area_m2=None,
#                                         nodata_mask=None,
#                                         normalize_weights=True,
#                                         out='km2'):
#     H, W, K = flow_quinn_cit.shape
#     assert K == 8, "Expected 8 neighbor weights (D8)."

#     if nodata_mask is None:
#         nodata_mask = np.zeros((H, W), dtype=bool)
#     else:
#         nodata_mask = nodata_mask.astype(bool, copy=False)

#     # 1) Sanitize weights
#     Wgt = np.nan_to_num(flow_quinn_cit.astype(np.float32, copy=True),
#                         nan=0.0, posinf=0.0, neginf=0.0)
#     Wgt = np.maximum(Wgt, 0.0)            # no negatives
#     Wgt[nodata_mask, :] = 0.0

#     # 2) Optional normalization (before indegree for consistency)
#     if normalize_weights:
#         sums = Wgt.sum(axis=2, dtype=np.float32)
#         pos = (sums > 0) & (~nodata_mask)
#         Wgt[pos, :] /= sums[pos, None]

#     D8 = [(-1, 1), (0, 1), (1, 1), (1, 0),
#           (1, -1), (0, -1), (-1, -1), (-1, 0)]

#     # 3) Init accumulation
#     if out == 'cells':
#         acc = np.ones((H, W), dtype=np.float64)
#     elif out in ('m2', 'km2'):
#         if pixel_area_m2 is None:
#             raise ValueError("pixel_area_m2 is required for out='m2' or 'km2'.")
#         acc = (np.full((H, W), float(pixel_area_m2), dtype=np.float64)
#                if np.isscalar(pixel_area_m2)
#                else pixel_area_m2.astype(np.float64, copy=True))
#     else:
#         raise ValueError("out must be one of {'cells','m2','km2'}")
#     acc[nodata_mask] = 0.0

#     # 4) In-degree
#     indeg = np.zeros((H, W), dtype=np.int32)
#     for i in range(H):
#         for j in range(W):
#             if nodata_mask[i, j]:
#                 continue
#             for k, (di, dj) in enumerate(D8):
#                 w = Wgt[i, j, k]
#                 if w <= 0.0:
#                     continue
#                 ni, nj = i + di, j + dj
#                 if 0 <= ni < H and 0 <= nj < W and not nodata_mask[ni, nj]:
#                     indeg[ni, nj] += 1

#     # 5) Kahn queue
#     from collections import deque
#     q = deque((i, j) for i in range(H) for j in range(W)
#               if (indeg[i, j] == 0 and not nodata_mask[i, j]))
#     visited = 0

#     while q:
#         i, j = q.popleft()
#         visited += 1
#         a_ij = acc[i, j]
#         for k, (di, dj) in enumerate(D8):
#             w = Wgt[i, j, k]
#             if w <= 0.0:
#                 continue
#             ni, nj = i + di, j + dj
#             if 0 <= ni < H and 0 <= nj < W and not nodata_mask[ni, nj]:
#                 acc[ni, nj] += a_ij * w
#                 indeg[ni, nj] -= 1
#                 if indeg[ni, nj] == 0:
#                     q.append((ni, nj))

#     # 6) Cycle check (should be 0 if flats were resolved)
#     if visited != np.count_nonzero(~nodata_mask):
#         raise RuntimeError("Cycle detected (unresolved flats?) – resolve flats with epsilon before accumulation.")

#     if out == 'km2':
#         acc *= 1e-6  # m^2 -> km^2

#     return acc.astype(np.float32)

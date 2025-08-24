# import numpy as np

# def compute_flow_accumulation_quinn_cit(flow_quinn_cit):
#     """
#     Compute flow accumulation based on Quinn 1995 with CIT flow direction weights.
    
#     Parameters:
#         flow_quinn_cit (3D ndarray): flow weights to 8 neighbors (rows x cols x 8)
    
#     Returns:
#         2D ndarray: accumulated area per cell (float32)
#     """
#     rows, cols, _ = flow_quinn_cit.shape
#     accumulation = np.ones((rows, cols), dtype=np.float32)  # each cell starts with 1
#     in_degree = np.zeros((rows, cols), dtype=np.int32)

#     # D8 neighbor offsets
#     D8 = [(-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0)]

#     # First pass: compute how many dependencies each cell has
#     for i in range(rows):
#         for j in range(cols):
#             for k, (di, dj) in enumerate(D8):
#                 ni, nj = i + di, j + dj
#                 if 0 <= ni < rows and 0 <= nj < cols:
#                     if flow_quinn_cit[i, j, k] > 0:
#                         in_degree[ni, nj] += 1

#     # Initialize processing queue with cells with zero in-degree
#     queue = [(i, j) for i in range(rows) for j in range(cols) if in_degree[i, j] == 0]

#     # Second pass: process topologically
#     while queue:
#         i, j = queue.pop(0)
#         for k, (di, dj) in enumerate(D8):
#             ni, nj = i + di, j + dj
#             if 0 <= ni < rows and 0 <= nj < cols:
#                 w = flow_quinn_cit[i, j, k]
#                 if w > 0:
#                     accumulation[ni, nj] += accumulation[i, j] * w
#                     in_degree[ni, nj] -= 1
#                     if in_degree[ni, nj] == 0:
#                         queue.append((ni, nj))

#     return accumulation


import numpy as np
from collections import deque

def compute_flow_accumulation_quinn_cit(flow_quinn_cit, pixel_area_m2, nodata_mask=None,
                                        normalize_weights=True, out='km2'):
    """
    Flow accumulation (Quinn 1995, CIT weights) with variable pixel area.
    
    Parameters:
        flow_quinn_cit : (H, W, 8) float32  -- flow fractions to 8 neighbors (sum<=1 per cell)
        pixel_area_m2  : (H, W) float32     -- per-pixel area in square meters (aligned grid)
        nodata_mask    : (H, W) bool or None -- True where cell is invalid (NoData)
        normalize_weights : bool             -- if True, renormalize positive weights to sum to 1
        out            : 'm2' or 'km2'       -- output units
    
    Returns:
        accumulation : (H, W) float32 in requested units
    """
    H, W, _ = flow_quinn_cit.shape
    assert pixel_area_m2.shape == (H, W)

    # Prepare masks
    if nodata_mask is None:
        nodata_mask = np.zeros((H, W), dtype=bool)

    # Copy weights and zero out invalid sources
    Wgt = flow_quinn_cit.astype(np.float32).copy()
    Wgt[nodata_mask, :] = 0.0

    # Optional normalization: for each valid cell, scale positive weights to sum to 1
    if normalize_weights:
        sums = Wgt.sum(axis=2, dtype=np.float32)
        pos = (sums > 0) & (~nodata_mask)
        Wgt[pos, :] /= sums[pos, None]

    # D8 neighbor offsets in the same order as weights
    D8 = np.array([(-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0)], dtype=np.int8)

    # Initialize accumulation with pixel area (each cell contributes its own area)
    acc = pixel_area_m2.astype(np.float64)  # use float64 for stability; cast back later
    acc[nodata_mask] = 0.0

    # Compute in-degree (number of upstream dependencies) per cell
    indeg = np.zeros((H, W), dtype=np.int32)

    for i in range(H):
        for j in range(W):
            if nodata_mask[i, j]:
                continue
            # outgoing edges from (i,j)
            for k, (di, dj) in enumerate(D8):
                if Wgt[i, j, k] <= 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and not nodata_mask[ni, nj]:
                    indeg[ni, nj] += 1

    # Topological processing queue: start with cells that have no incoming contributors
    q = deque([(i, j) for i in range(H) for j in range(W) if (indeg[i, j] == 0 and not nodata_mask[i, j])])

    while q:
        i, j = q.popleft()
        a_ij = acc[i, j]
        if a_ij == 0.0:
            # still need to decrement indegrees downstream
            pass
        for k, (di, dj) in enumerate(D8):
            w = Wgt[i, j, k]
            if w <= 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and not nodata_mask[ni, nj]:
                acc[ni, nj] += a_ij * w
                indeg[ni, nj] -= 1
                if indeg[ni, nj] == 0:
                    q.append((ni, nj))

    # Units
    if out == 'km2':
        acc = acc / 1e6
    elif out == 'm2':
        pass
    else:
        raise ValueError("out must be 'm2' or 'km2'")

    return acc.astype(np.float32)

import numpy as np
import heapq

# 8-neighborhood offsets
OFFS8 = [(-1,-1), (-1,0), (-1,1),
         ( 0,-1),         ( 0,1),
         ( 1,-1), ( 1,0), ( 1,1)]

def priority_flood_fill(
    dem: np.ndarray,
    nodata=np.nan,
    seed_internal_nodata_as_outlet=True,
    return_fill_depth=True
):
    """
    Priority-Flood (Barnes et al.) – full depression filling.
    - Flood from the raster boundary (outer border + optionally cells adjacent to internal NoData islands).
    - For floating-point grids: O(n log n) using a binary heap.
    - Output contains no depressions (every cell has an ensured outflow).

    Parameters
    ----------
    dem : 2D ndarray (float)
        Input DEM.
    nodata : number or np.nan
        NoData marker in the DEM.
    seed_internal_nodata_as_outlet : bool
        If True, treat internal NoData areas as outlets by seeding their valid neighbors into the priority queue.
        This is usually safer for real-world scenes containing masks/gaps.
    return_fill_depth : bool
        If True, also return the per-cell fill depth (how much elevation was added).

    Returns
    -------
    filled : 2D float64
        Depression-filled DEM.
    depth  : 2D float64 (optional)
        Amount of fill applied per cell (returned only if return_fill_depth=True).
    """
    Z = np.asarray(dem, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError("DEM must be 2D")
    nrows, ncols = Z.shape

    # valid-data mask
    valid = (np.isfinite(Z) if np.isnan(nodata) else ((Z != nodata) & np.isfinite(Z)))

    filled = Z.copy()
    visited = np.zeros_like(valid, dtype=bool)
    pq = []  # entries are (elevation, i, j)

    def inb(i, j): return (0 <= i < nrows) and (0 <= j < ncols)

    # 1) seed: all valid border cells of the raster
    for j in range(ncols):
        if valid[0, j] and not visited[0, j]:
            heapq.heappush(pq, (filled[0, j], 0, j)); visited[0, j] = True
        if valid[nrows-1, j] and not visited[nrows-1, j]:
            heapq.heappush(pq, (filled[nrows-1, j], nrows-1, j)); visited[nrows-1, j] = True
    for i in range(1, nrows-1):
        if valid[i, 0] and not visited[i, 0]:
            heapq.heappush(pq, (filled[i, 0], i, 0)); visited[i, 0] = True
        if valid[i, ncols-1] and not visited[i, ncols-1]:
            heapq.heappush(pq, (filled[i, ncols-1], i, ncols-1)); visited[i, ncols-1] = True

    # 2) optional seed: valid cells adjacent to NoData (treat internal NoData as outlet)
    if seed_internal_nodata_as_outlet:
        for i in range(nrows):
            for j in range(ncols):
                if valid[i, j] and not visited[i, j]:
                    for di, dj in OFFS8:
                        ni, nj = i + di, j + dj
                        if not inb(ni, nj) or not valid[ni, nj]:
                            heapq.heappush(pq, (filled[i, j], i, j))
                            visited[i, j] = True
                            break

    # 3) main loop: always expand the currently lowest water level
    while pq:
        elev, i, j = heapq.heappop(pq)
        for di, dj in OFFS8:
            ni, nj = i + di, j + dj
            if not inb(ni, nj) or not valid[ni, nj] or visited[ni, nj]:
                continue
            visited[ni, nj] = True
            # raise neighbor if it lies below the current water level
            if filled[ni, nj] < elev:
                filled[ni, nj] = elev
            heapq.heappush(pq, (filled[ni, nj], ni, nj))

    # outputs
    if np.isnan(nodata):
        filled[~valid] = np.nan
    else:
       filled[~valid] = float(nodata)

    if return_fill_depth:
        depth = (filled - Z)  # both float64
    if np.isnan(nodata):
        depth[~valid] = np.nan
    else:
        depth[~valid] = 0.0
    return filled, depth

    return filled

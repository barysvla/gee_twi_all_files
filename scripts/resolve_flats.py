import numpy as np
from collections import deque

def resolve_flats_towards_lower_edge(dem, nodata=np.nan, epsilon=1e-4):
    """
    Resolve flat areas (plateaus) in a DEM by imposing a tiny gradient
    towards the lower edge of each flat region (Garbrecht & Martz style).
    
    Parameters
    ----------
    dem : np.ndarray (2D, float)
        Digital elevation model.
    nodata : float or np.nan
        NoData value. Cells equal to nodata are ignored/untouched.
    epsilon : float
        Small elevation increment to impose per 'step' inside flats.
        Keep it several orders below natural DEM noise/resolution.
    
    Returns
    -------
    dem_out : np.ndarray (2D, float)
        DEM with flats resolved (tiny gradients added inside flats).
    """

    dem = np.asarray(dem)
    if dem.ndim != 2:
        raise ValueError("DEM must be a 2D array")

    nrows, ncols = dem.shape
    dem_out = dem.copy()

    # --- masks ---
    if np.isnan(nodata):
        valid = ~np.isnan(dem)
    else:
        valid = dem != nodata

    # neighbor offsets (8-neighborhood)
    OFFS = [(-1,-1), (-1,0), (-1,1),
            ( 0,-1),         ( 0,1),
            ( 1,-1), ( 1,0), ( 1,1)]

    def in_bounds(i, j):
        return 0 <= i < nrows and 0 <= j < ncols

    # --- helper: compute neighbor relations for all cells ---
    # For each cell: has_lower, has_equal, has_higher among valid neighbors
    has_lower  = np.zeros_like(valid, dtype=bool)
    has_equal  = np.zeros_like(valid, dtype=bool)
    has_higher = np.zeros_like(valid, dtype=bool)

    for di, dj in OFFS:
        ni = np.clip(np.arange(nrows)[:,None] + di, 0, nrows-1)
        nj = np.clip(np.arange(ncols)[None,:] + dj, 0, ncols-1)
        # Using rolled views would wrap; instead we map to clipped neighbors
        # and mask out cells where neighbor is actually out of bounds.
        # We'll exclude true out-of-bounds by separate mask:
        mask_nb = np.zeros_like(valid, dtype=bool)
        mask_nb[max(0,di):nrows+min(0,di), max(0,dj):ncols+min(0,dj)] = True

        nb_vals = dem[ni, nj]
        nb_valid = valid & mask_nb & valid[ni, nj]

        has_lower  |= nb_valid & (nb_vals < dem)
        has_equal  |= nb_valid & (nb_vals == dem)
        has_higher |= nb_valid & (nb_vals > dem)

    # --- identify flat candidate cells ---
    # Flat cells: valid, have at least one equal neighbor, and NO strictly lower neighbor.
    flats_mask = valid & has_equal & (~has_lower)

    # --- label connected flat regions with simple flood-fill (8-neighborhood) ---
    labels = np.full(dem.shape, -1, dtype=int)
    current_label = 0

    for i in range(nrows):
        for j in range(ncols):
            if flats_mask[i, j] and labels[i, j] == -1:
                # BFS to label the flat region
                q = deque()
                q.append((i, j))
                labels[i, j] = current_label
                z0 = dem[i, j]
                while q:
                    ci, cj = q.popleft()
                    for di, dj in OFFS:
                        ni, nj = ci + di, cj + dj
                        if in_bounds(ni, nj) and labels[ni, nj] == -1:
                            if flats_mask[ni, nj] and dem[ni, nj] == z0:
                                labels[ni, nj] = current_label
                                q.append((ni, nj))
                current_label += 1

    if current_label == 0:
        # No flats -> nothing to do
        return dem_out

    # --- helper BFS distance inside a region from a frontier set ---
    def bfs_distance_from_frontier(region_mask, frontier_mask):
        """
        Compute integer distance (in 8-neighborhood steps) from the frontier cells
        to all cells within region_mask. Frontier cells have distance 0.
        Non-region cells return -1.
        """
        dist = np.full(region_mask.shape, -1, dtype=int)
        q = deque()

        # initialize with frontier
        fi, fj = np.where(frontier_mask & region_mask)
        for a, b in zip(fi, fj):
            dist[a, b] = 0
            q.append((a, b))

        # BFS
        while q:
            ci, cj = q.popleft()
            for di, dj in OFFS:
                ni, nj = ci + di, cj + dj
                if in_bounds(ni, nj) and region_mask[ni, nj] and dist[ni, nj] == -1:
                    dist[ni, nj] = dist[ci, cj] + 1
                    q.append((ni, nj))
        return dist

    # --- process each flat region independently ---
    for lbl in range(current_label):
        region = labels == lbl
        if not np.any(region):
            continue

        # Identify "lower edge" cells: flat cells adjacent to any lower neighbor outside flat
        lower_edge = np.zeros_like(region, dtype=bool)
        # Identify "upper edge" cells: flat cells adjacent to any higher neighbor outside flat
        upper_edge = np.zeros_like(region, dtype=bool)

        # Also keep the constant elevation of this flat
        flat_z = dem[region][0]

        # For each cell in region, check neighbors
        ri, rj = np.where(region)
        for ci, cj in zip(ri, rj):
            for di, dj in OFFS:
                ni, nj = ci + di, cj + dj
                if not in_bounds(ni, nj) or not valid[ni, nj]:
                    # touching NoData or boundary with no valid lower? Treat as not an outlet.
                    continue
                if not region[ni, nj]:
                    if dem[ni, nj] < flat_z:
                        lower_edge[ci, cj] = True
                    if dem[ni, nj] > flat_z:
                        upper_edge[ci, cj] = True

        # If there is no lower edge, we cannot drain this flat; skip safely.
        if not np.any(lower_edge):
            continue

        # Distance field 1: from lower edge inward (primary gradient driver)
        dist_down = bfs_distance_from_frontier(region, lower_edge)
        # Distance field 2: from upper edge inward (tie-breaker; if none, use zeros)
        if np.any(upper_edge):
            dist_up = bfs_distance_from_frontier(region, upper_edge)
            # Convert to "reverse" (farther from upper edge -> larger reverse)
            max_up = dist_up[region].max()
            dist_up_rev = np.zeros_like(dist_up, dtype=float)
            # Avoid -1 (unreached) by setting to 0
            valid_up = (dist_up >= 0) & region
            dist_up_rev[valid_up] = (max_up - dist_up[valid_up]).astype(float)
        else:
            dist_up = np.zeros_like(region, dtype=int)
            dist_up_rev = np.zeros_like(region, dtype=float)

        # Compose tiny adjustment:
        #  - Primary: epsilon * dist_down     (ensures flow towards lower edge)
        #  - Secondary: epsilon * 1e-3 * dist_up_rev (breaks ties toward upper boundary)
        adjust = np.zeros_like(dem_out, dtype=float)
        mask_ok = (dist_down >= 0) & region
        adjust[mask_ok] = epsilon * (dist_down[mask_ok].astype(float) +
                                     1e-3 * dist_up_rev[mask_ok])

        # Apply adjustment to this flat region
        dem_out[region] = dem_out[region] + adjust[region]

    return dem_out

import numpy as np
from collections import deque

def resolve_flats_towards_lower_edge(
    dem,
    nodata=np.nan,
    epsilon=1e-4,
    equal_tol=1e-3,
    equality_connectivity=8,
    treat_oob_as_lower=True
):
    """
    Garbrecht & Martz styl: vynutí minimální sklon TOWARDS nižší okraj plošiny.
    Vrací i flatmask = integer hodnoty posunu uvnitř plošin.
    """
    Z = np.asarray(dem)
    if Z.ndim != 2:
        raise ValueError("DEM must be 2D")
    nrows, ncols = Z.shape

    if np.isnan(nodata):
        valid = np.isfinite(Z)
    else:
        valid = (Z != nodata) & np.isfinite(Z)

    OFFS8 = [(-1,-1), (-1,0), (-1,1),
             ( 0,-1),         ( 0,1),
             ( 1,-1), ( 1,0), ( 1,1)]
    OFFS4 = [(-1,0), (1,0), (0,-1), (0,1)]
    OFFS_EQ = OFFS4 if equality_connectivity == 4 else OFFS8

    def inb(i,j): return 0 <= i < nrows and 0 <= j < ncols

    # --- kdo NEMÁ nižšího souseda ---
    has_lower = np.zeros_like(valid, bool)
    for di,dj in OFFS8:
        i0, i1 = max(0,-di), min(nrows, nrows-di)
        j0, j1 = max(0,-dj), min(ncols, ncols-dj)
        if i0>=i1 or j0>=j1: 
            continue
        a = Z[i0:i1, j0:j1]
        b = Z[i0+di:i1+di, j0+dj:j1+dj]
        v = valid[i0:i1, j0:j1] & valid[i0+di:i1+di, j0+dj:j1+dj]
        has_lower[i0:i1, j0:j1] |= v & ((b - a) < -equal_tol)

    flats_mask = valid & (~has_lower)

    # --- labelování ---
    labels = np.full(Z.shape, -1, np.int32)
    cur = 0
    for i in range(nrows):
        for j in range(ncols):
            if flats_mask[i,j] and labels[i,j] == -1:
                z0 = Z[i,j]
                q = deque([(i,j)])
                labels[i,j] = cur
                while q:
                    ci,cj = q.popleft()
                    for di,dj in OFFS_EQ:
                        ni,nj = ci+di, cj+dj
                        if (inb(ni,nj) and flats_mask[ni,nj] and labels[ni,nj]==-1
                            and abs(Z[ni,nj]-z0) <= equal_tol):
                            labels[ni,nj] = cur
                            q.append((ni,nj))
                cur += 1

    # --- BFS dist od frontier ---
    def bfs_distance(region_mask, frontier_mask):
        dist = np.full(region_mask.shape, -1, np.int32)
        q = deque()
        fi, fj = np.where(frontier_mask & region_mask)
        for a,b in zip(fi,fj):
            dist[a,b] = 0
            q.append((a,b))
        while q:
            ci,cj = q.popleft()
            d0 = dist[ci,cj]
            for di,dj in OFFS8:
                ni,nj = ci+di, cj+dj
                if inb(ni,nj) and region_mask[ni,nj] and dist[ni,nj] == -1:
                    dist[ni,nj] = d0 + 1
                    q.append((ni,nj))
        return dist

    dem_out = Z.astype(np.float32, copy=True)
    flatmask = np.zeros_like(Z, np.int32)
    n_changed = 0

    for lbl in range(cur):
        region = (labels == lbl)
        if not np.any(region):
            continue
        flat_z = Z[region][0]

        lower_edge = np.zeros_like(region, bool)
        upper_edge = np.zeros_like(region, bool)

        ri, rj = np.where(region)
        for ci,cj in zip(ri, rj):
            for di,dj in OFFS8:
                ni,nj = ci+di, cj+dj
                if not inb(ni,nj) or not valid[ni,nj]:
                    if treat_oob_as_lower:
                        lower_edge[ci,cj] = True
                    continue
                if not region[ni,nj]:
                    dz = Z[ni,nj] - flat_z
                    if dz < -equal_tol:
                        lower_edge[ci,cj] = True
                    elif dz >  equal_tol:
                        upper_edge[ci,cj] = True

        if not np.any(lower_edge):
            continue

        dist_down = bfs_distance(region, lower_edge)

        if np.any(upper_edge):
            dist_up = bfs_distance(region, upper_edge)
            max_up = dist_up[region].max()
            dist_up_rev = np.zeros_like(dist_up, dtype=float)
            sel_up = (dist_up >= 0) & region
            if max_up > 0:
                dist_up_rev[sel_up] = (max_up - dist_up[sel_up]).astype(float)
        else:
            dist_up_rev = np.zeros_like(Z, dtype=float)

        sel = region & (dist_down >= 0)
        if np.any(sel):
            # integer mask pro debugování
            flatmask[sel] = dist_down[sel]
            # finální posun
            delta = epsilon * (dist_down[sel].astype(float) + 1e-3*dist_up_rev[sel])
            dem_out[sel] = dem_out[sel] + delta.astype(np.float32)
            n_changed += int(delta.size)

    dem_out[~valid] = (np.nan if np.isnan(nodata) else nodata)

    stats = {
        "n_flats": int(cur),
        "n_flat_cells": int(np.count_nonzero(labels >= 0)),
        "n_changed_cells": int(n_changed),
        "equal_tol": float(equal_tol),
        "equality_connectivity": int(equality_connectivity),
        "treat_oob_as_lower": bool(treat_oob_as_lower),
    }
    return dem_out, flatmask, labels, stats

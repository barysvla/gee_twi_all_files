import numpy as np
from collections import deque

# najde 4500 bunek, ale acc nefunguje
def resolve_flats_towards_lower_edge(
    dem,
    nodata=np.nan,
    epsilon=1e-5,             # řádově jako pysheds
    equal_tol=1e-3,           # pro rovnost (labeling)
    lower_tol=0.0,            # pro "nižší" souseda (detekce odtoku)
    equality_connectivity=8,  # 4/8 pro spojování plošin
    treat_oob_as_lower=True,  # OOB/NoData jako "lower"
    force_all_flats=False,    # změnit i plošiny bez skutečného lower edge (perimetr)
    bump_frontier=True        # změnit i frontier (dist==0) -> jako u tvojeho Barnes
):
    Z = np.asarray(dem)
    if Z.ndim != 2:
        raise ValueError("DEM must be 2D")
    nrows, ncols = Z.shape

    valid = np.isfinite(Z) if np.isnan(nodata) else ((Z != nodata) & np.isfinite(Z))

    OFFS8 = [(-1,-1), (-1,0), (-1,1),
             ( 0,-1),         ( 0,1),
             ( 1,-1), ( 1,0), ( 1,1)]
    OFFS4 = [(-1,0), (1,0), (0,-1), (0,1)]
    OFFS_EQ = OFFS4 if equality_connectivity == 4 else OFFS8

    def inb(i,j): return 0 <= i < nrows and 0 <= j < ncols

    # --- 1) "má nižšího souseda?" (8-conn, s lower_tol)
    has_lower = np.zeros_like(valid, bool)
    for di,dj in OFFS8:
        i0, i1 = max(0,-di), min(nrows, nrows-di)
        j0, j1 = max(0,-dj), min(ncols, ncols-dj)
        if i0>=i1 or j0>=j1: 
            continue
        a = Z[i0:i1, j0:j1]
        b = Z[i0+di:i1+di, j0+dj:j1+dj]
        v = valid[i0:i1, j0:j1] & valid[i0+di:i1+di, j0+dj:j1+dj]
        has_lower[i0:i1, j0:j1] |= v & ((b - a) < -lower_tol)

    # Flats: NEMÁ nižšího souseda
    flats_mask = valid & (~has_lower)

    # --- 2) Labeling plošin (± equal_tol, 4/8-conn)
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
                        if (inb(ni,nj) and flats_mask[ni,nj]
                            and labels[ni,nj]==-1
                            and abs(Z[ni,nj]-z0) <= equal_tol):
                            labels[ni,nj] = cur
                            q.append((ni,nj))
                cur += 1

    # --- BFS vzdálenost od fronty (8-conn)
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
    n_drainable = 0
    n_forced = 0

    # --- 3) Vynutit sklon na všech plošinách (Barnes-like chování)
    for lbl in range(cur):
        region = (labels == lbl)
        if not np.any(region):
            continue
        flat_z = Z[region][0]

        # a) "nižší okraj" + kaskáda: rovný soused mimo plošinu, který SÁM má nižšího souseda
        lower_edge = np.zeros_like(region, bool)
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
                    if dz < -lower_tol:
                        lower_edge[ci,cj] = True
                    elif abs(dz) <= equal_tol and has_lower[ni,nj]:
                        lower_edge[ci,cj] = True  # "kaskádový" outlet

        use_fallback = False
        if np.any(lower_edge):
            frontier = lower_edge
            n_drainable += 1
        else:
            if not force_all_flats:
                continue
            # fallback: perimetr plošiny
            perim = np.zeros_like(region, bool)
            for ci,cj in zip(ri, rj):
                for di,dj in OFFS8:
                    ni,nj = ci+di, cj+dj
                    if not inb(ni,nj) or not region[ni,nj]:
                        perim[ci,cj] = True
                        break
            frontier = perim
            use_fallback = True
            n_forced += 1

        dist_down = bfs_distance(region, frontier)

        sel = region & (dist_down >= 0)
        if not np.any(sel):
            continue

        # --- zásah: bumpnout i frontier pro stejné chování jako u Barnes
        eff_dist = dist_down[sel] + (1 if bump_frontier else 0)
        flatmask[sel] = dist_down[sel]
        delta = (epsilon * eff_dist).astype(np.float32)
        dem_out[sel] = dem_out[sel] + delta
        n_changed += int(sel.sum()) if bump_frontier else int(np.count_nonzero(dist_down[sel] > 0))

    # NoData zpět
    dem_out[~valid] = (np.nan if np.isnan(nodata) else nodata)

    stats = {
        "n_flats": int(cur),
        "n_flat_cells": int(np.count_nonzero(labels >= 0)),
        "n_changed_cells": int(n_changed),
        "n_flats_drainable": int(n_drainable),
        "n_flats_forced": int(n_forced),
        "equal_tol": float(equal_tol),
        "lower_tol": float(lower_tol),
        "equality_connectivity": int(equality_connectivity),
        "treat_oob_as_lower": bool(treat_oob_as_lower),
        "force_all_flats": bool(force_all_flats),
        "bump_frontier": bool(bump_frontier),
        "epsilon": float(epsilon),
    }
    return dem_out, flatmask, labels, stats

# -------------------------------------------------------
import numpy as np
from collections import deque

# nejlip funguje
def resolve_flats_barnes(
    dem,
    nodata=np.nan,
    epsilon=1e-5,
    equal_tol=1e-3,           # tolerance pro rovnost (spojování plošin)
    lower_tol=0.0,            # tolerance pro "nižší" (doporuč. 0..1e-6)
    treat_oob_as_lower=True,
    require_low_edge_only=True,
    force_all_flats=False     # vynutit řešení i bez výtoku
):
    Z = np.asarray(dem)
    if Z.ndim != 2:
        raise ValueError("DEM must be 2D")
    nrows, ncols = Z.shape

    valid = np.isfinite(Z) if np.isnan(nodata) else ((Z != nodata) & np.isfinite(Z))

    OFFS8 = [(-1,-1),(-1,0),(-1,1),
             ( 0,-1),       ( 0,1),
             ( 1,-1),( 1,0),( 1,1)]
    def inb(i,j): return 0 <= i < nrows and 0 <= j < ncols

    # --- 1) předpočítej "má nižšího souseda" s prahováním lower_tol ---
    has_lower8 = np.zeros_like(valid, bool)
    for di,dj in OFFS8:
        i0,i1 = max(0,-di), min(nrows, nrows-di)
        j0,j1 = max(0,-dj), min(ncols, ncols-dj)
        if i0>=i1 or j0>=j1: continue
        a = Z[i0:i1, j0:j1]
        b = Z[i0+di:i1+di, j0+dj:j1+dj]
        v = valid[i0:i1, j0:j1] & valid[i0+di:i1+di, j0+dj:j1+dj]
        has_lower8[i0:i1, j0:j1] |= v & ((b - a) < -lower_tol)

    # flats: NEMÁ nižšího souseda (rovnost se nevyžaduje)
    flats = valid & (~has_lower8)

    # --- 2) labelování plošin (8-conn, s rovnostní tolerancí equal_tol) ---
    labels = np.zeros_like(valid, np.int32)
    cur = 0
    for i in range(nrows):
        for j in range(ncols):
            if flats[i,j] and labels[i,j]==0:
                cur += 1
                q = deque([(i,j)])
                labels[i,j] = cur
                z0 = Z[i,j]
                while q:
                    ci,cj = q.popleft()
                    for di,dj in OFFS8:
                        ni,nj = ci+di, cj+dj
                        if inb(ni,nj) and flats[ni,nj] and labels[ni,nj]==0 and abs(Z[ni,nj]-z0) <= equal_tol:
                            labels[ni,nj] = cur
                            q.append((ni,nj))

    if cur == 0:
        dem_out = Z.astype(np.float32, copy=True)
        dem_out[~valid] = (np.nan if np.isnan(nodata) else nodata)
        stats = {"n_flats":0,"n_flat_cells":0,"n_changed_cells":0,"n_flats_drainable":0}
        return dem_out, np.zeros_like(labels), labels, stats

    # --- 3) hrany; "kaskádový low-edge" přes rovné sousedy s has_lower8=True ---
    HighEdges = [deque() for _ in range(cur+1)]
    LowEdges  = [deque() for _ in range(cur+1)]
    has_low   = np.zeros(cur+1, bool)
    has_high  = np.zeros(cur+1, bool)

    for i in range(nrows):
        for j in range(ncols):
            lbl = labels[i,j]
            if lbl==0: continue
            z0 = Z[i,j]
            adj_higher = False
            adj_lower  = False
            is_boundary = False
            for di,dj in OFFS8:
                ni,nj = i+di, j+dj
                if not inb(ni,nj) or not valid[ni,nj]:
                    is_boundary = True
                    if treat_oob_as_lower:
                        adj_lower = True
                    continue
                if labels[ni,nj] != lbl:
                    dz = Z[ni,nj] - z0
                    if dz >  equal_tol: adj_higher = True
                    # přísně nižší
                    if dz < -lower_tol: adj_lower  = True
                    # kaskáda: soused "rovný" a sám má nižšího souseda
                    elif abs(dz) <= equal_tol and has_lower8[ni,nj]:
                        adj_lower = True
            if adj_lower:
                LowEdges[lbl].append((i,j)); has_low[lbl]=True
            if adj_higher:
                HighEdges[lbl].append((i,j)); has_high[lbl]=True
            # optional fallback: když force_all_flats a plošina je úplně "uvnitř",
            # budeme jako seeds pro BFS brát aspoň její hranové buňky
            if force_all_flats and is_boundary and not adj_lower:
                LowEdges[lbl].append((i,j))  # seed, i když není skutečně "lower"
                has_low[lbl] = True

    def flat_active(lbl):
        if has_low[lbl]:
            return True
        return force_all_flats  # vynutit řešení i bez skutečného výtoku

    # --- 4) dvě BFS (mark-on-enqueue) ---
    away     = np.full(labels.shape, -1, np.int32)
    towards  = np.full(labels.shape, -1, np.int32)
    FlatMask = np.zeros_like(labels, np.int32)
    FlatH    = np.zeros(cur+1, np.int32)

    # A) od vyššího
    for lbl in range(1, cur+1):
        if not flat_active(lbl): continue
        q = HighEdges[lbl]
        if not q: continue
        for si,sj in q: away[si,sj] = 1
        while q:
            ci,cj = q.popleft()
            cl = away[ci,cj]
            if cl > FlatH[lbl]: FlatH[lbl] = cl
            for di,dj in OFFS8:
                ni,nj = ci+di, cj+dj
                if inb(ni,nj) and labels[ni,nj]==lbl and away[ni,nj]==-1:
                    away[ni,nj] = cl + 1
                    q.append((ni,nj))

    # B) k nižšímu (dominantní 2*dist) + kombinace
    drainable = 0
    for lbl in range(1, cur+1):
        if not flat_active(lbl): continue
        q = LowEdges[lbl]
        if not q: continue
        if has_low[lbl]: drainable += 1
        for si,sj in q: towards[si,sj] = 1
        while q:
            ci,cj = q.popleft()
            cl = towards[ci,cj]
            if away[ci,cj] != -1:
                FlatMask[ci,cj] = (FlatH[lbl] - away[ci,cj]) + 2*cl
            else:
                FlatMask[ci,cj] = 2*cl
            for di,dj in OFFS8:
                ni,nj = ci+di, cj+dj
                if inb(ni,nj) and labels[ni,nj]==lbl and towards[ni,nj]==-1:
                    towards[ni,nj] = cl + 1
                    q.append((ni,nj))

    # --- 5) aplikace ---
    dem_out = Z.astype(np.float32, copy=True)
    inc = (labels>0) & valid
    dem_out[inc] = dem_out[inc] + epsilon * FlatMask[inc]

    if np.isnan(nodata): dem_out[~valid] = np.nan
    else:                dem_out[~valid] = nodata

    stats = {
        "n_flats": int(cur),
        "n_flats_drainable": int(drainable),
        "n_flat_cells": int(np.count_nonzero(labels)),
        "n_changed_cells": int(np.count_nonzero((FlatMask!=0) & inc)),
        "equal_tol": float(equal_tol),
        "lower_tol": float(lower_tol),
        "force_all_flats": bool(force_all_flats)
    }
    return dem_out, FlatMask.astype(np.int32), labels, stats

# -------------------------------------------------------------------
# TIE MODIFIKACE- dava zmenenych 10 k ale...
# import numpy as np
# from collections import deque

# def resolve_flats_barnes(
#     dem,
#     nodata=np.nan,
#     epsilon=1e-5,           # sníženo o řád (kvůli rozsahu Δ)
#     equal_tol=1e-3,
#     lower_tol=0.0,
#     treat_oob_as_lower=True,
#     require_low_edge_only=True,
#     force_all_flats=False,
#     include_equal_ties=True  # <<< NOVÉ: zapnout „tie“ buňky (jako pysheds)
# ):
#     Z = np.asarray(dem)
#     if Z.ndim != 2:
#         raise ValueError("DEM must be 2D")
#     nrows, ncols = Z.shape

#     valid = np.isfinite(Z) if np.isnan(nodata) else ((Z != nodata) & np.isfinite(Z))

#     OFFS8 = [(-1,-1),(-1,0),(-1,1),
#              ( 0,-1),       ( 0,1),
#              ( 1,-1),( 1,0),( 1,1)]
#     def inb(i,j): return 0 <= i < nrows and 0 <= j < ncols

#     # --- "má nižšího souseda?" (8-conn, s prahem lower_tol) ---
#     has_lower8 = np.zeros_like(valid, bool)
#     for di,dj in OFFS8:
#         i0,i1 = max(0,-di), min(nrows, nrows-di)
#         j0,j1 = max(0,-dj), min(ncols, ncols-dj)
#         if i0>=i1 or j0>=j1: continue
#         a = Z[i0:i1, j0:j1]
#         b = Z[i0+di:i1+di, j0+dj:j1+dj]
#         v = valid[i0:i1, j0:j1] & valid[i0+di:i1+di, j0+dj:j1+dj]
#         has_lower8[i0:i1, j0:j1] |= v & ((b - a) < -lower_tol)

#     # kandidáti na "flats" (bez přísně nižšího souseda)
#     flats = valid & (~has_lower8)

#     # --- labelování (8-conn) ---
#     labels = np.zeros_like(valid, np.int32)
#     cur = 0
#     for i in range(nrows):
#         for j in range(ncols):
#             if flats[i,j] and labels[i,j]==0:
#                 cur += 1
#                 q = deque([(i,j)])
#                 labels[i,j] = cur
#                 z0 = Z[i,j]
#                 while q:
#                     ci,cj = q.popleft()
#                     for di,dj in OFFS8:
#                         ni,nj = ci+di, cj+dj
#                         if (inb(ni,nj) and flats[ni,nj] and labels[ni,nj]==0
#                             and abs(Z[ni,nj]-z0) <= equal_tol):
#                             labels[ni,nj] = cur
#                             q.append((ni,nj))

#     # --- AUGMENTACE: připojit "tie" buňky (rovné ± equal_tol), i když nejsou flats ---
#     if include_equal_ties and cur > 0:
#         for lbl in range(1, cur+1):
#             region = (labels == lbl)
#             if not np.any(region): 
#                 continue
#             z0 = Z[region][0]
#             q = deque(list(zip(*np.where(region))))
#             while q:
#                 ci,cj = q.popleft()
#                 for di,dj in OFFS8:
#                     ni,nj = ci+di, cj+dj
#                     if inb(ni,nj) and valid[ni,nj] and labels[ni,nj]==0:
#                         if abs(Z[ni,nj]-z0) <= equal_tol:
#                             labels[ni,nj] = lbl
#                             q.append((ni,nj))

#     if cur == 0:
#         dem_out = Z.astype(np.float32, copy=True)
#         dem_out[~valid] = (np.nan if np.isnan(nodata) else nodata)
#         stats = {"n_flats":0,"n_flat_cells":0,"n_changed_cells":0,"n_flats_drainable":0}
#         return dem_out, np.zeros_like(labels), labels, stats

#     # --- hrany (kaskáda: rovný soused, který sám má nižšího) ---
#     HighEdges = [deque() for _ in range(cur+1)]
#     LowEdges  = [deque() for _ in range(cur+1)]
#     has_low   = np.zeros(cur+1, bool)
#     has_high  = np.zeros(cur+1, bool)

#     for i in range(nrows):
#         for j in range(ncols):
#             lbl = labels[i,j]
#             if lbl==0: continue
#             z0 = Z[i,j]
#             adj_higher = False
#             adj_lower  = False
#             is_boundary = False
#             for di,dj in OFFS8:
#                 ni,nj = i+di, j+dj
#                 if not inb(ni,nj) or not valid[ni,nj]:
#                     is_boundary = True
#                     if treat_oob_as_lower:
#                         adj_lower = True
#                     continue
#                 if labels[ni,nj] != lbl:
#                     dz = Z[ni,nj] - z0
#                     if dz >  equal_tol: adj_higher = True
#                     if dz < -lower_tol: adj_lower  = True
#                     elif abs(dz) <= equal_tol and has_lower8[ni,nj]:
#                         adj_lower = True
#             if adj_lower:
#                 LowEdges[lbl].append((i,j)); has_low[lbl]=True
#             if adj_higher:
#                 HighEdges[lbl].append((i,j)); has_high[lbl]=True
#             if force_all_flats and is_boundary and not adj_lower:
#                 LowEdges[lbl].append((i,j)); has_low[lbl]=True

#     def flat_active(lbl):
#         return has_low[lbl] or force_all_flats

#     # --- dvě BFS, mark-on-enqueue ---
#     away     = np.full(labels.shape, -1, np.int32)
#     towards  = np.full(labels.shape, -1, np.int32)
#     FlatMask = np.zeros_like(labels, np.int32)
#     FlatH    = np.zeros(cur+1, np.int32)

#     # A) od vyššího
#     for lbl in range(1, cur+1):
#         if not flat_active(lbl): continue
#         q = HighEdges[lbl]
#         if not q: continue
#         for si,sj in q: away[si,sj] = 1
#         while q:
#             ci,cj = q.popleft()
#             cl = away[ci,cj]
#             if cl > FlatH[lbl]: FlatH[lbl] = cl
#             for di,dj in OFFS8:
#                 ni,nj = ci+di, cj+dj
#                 if inb(ni,nj) and labels[ni,nj]==lbl and away[ni,nj]==-1:
#                     away[ni,nj] = cl + 1
#                     q.append((ni,nj))

#     # B) k nižšímu + kombinace
#     drainable = 0
#     for lbl in range(1, cur+1):
#         if not flat_active(lbl): continue
#         q = LowEdges[lbl]
#         if not q: continue
#         if has_low[lbl]: drainable += 1
#         for si,sj in q: towards[si,sj] = 1
#         while q:
#             ci,cj = q.popleft()
#             cl = towards[ci,cj]
#             if away[ci,cj] != -1:
#                 FlatMask[ci,cj] = (FlatH[lbl] - away[ci,cj]) + 2*cl
#             else:
#                 FlatMask[ci,cj] = 2*cl
#             for di,dj in OFFS8:
#                 ni,nj = ci+di, cj+dj
#                 if inb(ni,nj) and labels[ni,nj]==lbl and towards[ni,nj]==-1:
#                     towards[ni,nj] = cl + 1
#                     q.append((ni,nj))

#     # --- aplikace Δ ---
#     dem_out = Z.astype(np.float32, copy=True)
#     inc = (labels>0) & valid
#     dem_out[inc] = dem_out[inc] + epsilon * FlatMask[inc]

#     if np.isnan(nodata): dem_out[~valid] = np.nan
#     else:                dem_out[~valid] = nodata

#     stats = {
#         "n_flats": int(cur),
#         "n_flats_drainable": int(drainable),
#         "n_flat_cells": int(np.count_nonzero(labels)),
#         "n_changed_cells": int(np.count_nonzero((FlatMask!=0) & inc)),
#         "equal_tol": float(equal_tol),
#         "lower_tol": float(lower_tol),
#         "include_equal_ties": bool(include_equal_ties),
#         "epsilon": float(epsilon),
#     }
#     return dem_out, FlatMask.astype(np.int32), labels, stats

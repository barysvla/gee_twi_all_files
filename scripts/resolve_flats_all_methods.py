from __future__ import annotations

import numpy as np
from collections import deque
from typing import Dict, Tuple, Optional


import numpy as np
from collections import deque

# D8 neighbor offsets (index 0..7)
D8_OFFS = [(-1, -1), (-1, 0), (-1, 1),
           ( 0, -1),          ( 0, 1),
           ( 1, -1), ( 1, 0), ( 1, 1)]

NOFLOW = -1     # "NoFlow" in the paper
NODATA_FD = -999
FLOW_OUT = 8    # synthetic direction "out of DEM" for boundary cells


def _inb(i: int, j: int, nrows: int, ncols: int) -> bool:
    return 0 <= i < nrows and 0 <= j < ncols


def _compute_flowdirs_d8_min_neighbor(
    dem: np.ndarray,
    valid: np.ndarray,
    *,
    boundary_flows_out: bool = True
) -> np.ndarray:
    """
    Compute D8 flow directions.
    - If a cell has at least one strictly-lower neighbor, flow goes to the lowest neighbor.
    - Otherwise, flow direction is NOFLOW.
    - If boundary_flows_out=True, valid boundary cells are assigned FLOW_OUT.
    """
    nrows, ncols = dem.shape
    fd = np.full((nrows, ncols), NODATA_FD, dtype=np.int16)

    for i in range(nrows):
        for j in range(ncols):
            if not valid[i, j]:
                continue

            # Boundary assumption from the paper: edges can flow out of DEM
            if boundary_flows_out and (i == 0 or j == 0 or i == nrows - 1 or j == ncols - 1):
                fd[i, j] = FLOW_OUT
                continue

            z0 = dem[i, j]
            best_k = NOFLOW
            best_z = z0

            for k, (di, dj) in enumerate(D8_OFFS):
                ni, nj = i + di, j + dj
                if not _inb(ni, nj, nrows, ncols) or not valid[ni, nj]:
                    continue
                z1 = dem[ni, nj]
                if z1 < best_z:
                    best_z = z1
                    best_k = k

            fd[i, j] = best_k  # NOFLOW if none lower
    return fd


def resolve_flats_barnes_2014(
    dem: np.ndarray,
    nodata: float = np.nan,
    *,
    boundary_flows_out: bool = True,
    apply_to_dem: bool = False,
    epsilon: float = 2e-5
):
    """
    Barnes (2014) flat resolution:
    - Step 0: compute FlowDirections (cells with no lower neighbor get NOFLOW)
    - Step 1: find HighEdges and LowEdges, label drainable flats via flood-fill from LowEdges,
              remove undrainable entries from HighEdges
    - Step 2: BFS away-from-higher -> store distances in FlatMask (and max per flat in FlatHeights)
    - Step 3: BFS towards-lower and combine with Step 2; towards-lower has double weight
    - Output: FlatMask + Labels; optionally apply epsilon*FlatMask to DEM

    Returns:
      dem_out  : DEM (unchanged unless apply_to_dem=True)
      flatmask : int32 FlatMask
      labels   : int32 Labels (0 = not a drainable flat)
      stats    : dict
    """
    Z = np.asarray(dem)
    if Z.ndim != 2:
        raise ValueError("DEM must be 2D")

    Zf = Z.astype(np.float64, copy=False)
    nrows, ncols = Zf.shape

    # Valid mask
    if np.isnan(nodata):
        valid = np.isfinite(Zf)
    else:
        valid = (Zf != nodata) & np.isfinite(Zf)

    # ---- Step 0: initial flow directions (example as in the paper) ----
    flow = _compute_flowdirs_d8_min_neighbor(Zf, valid, boundary_flows_out=boundary_flows_out)

    # ---- Step 1a: build HighEdges and LowEdges (paper definition) ----
    high_edges = deque()
    low_edges = deque()

    for i in range(nrows):
        for j in range(ncols):
            if not valid[i, j]:
                continue

            z0 = Zf[i, j]
            fd0 = flow[i, j]

            # High edge: (1) NOFLOW and (2) has a higher neighbor
            if fd0 == NOFLOW:
                has_higher = False
                for di, dj in D8_OFFS:
                    ni, nj = i + di, j + dj
                    if not _inb(ni, nj, nrows, ncols) or not valid[ni, nj]:
                        continue
                    if Zf[ni, nj] > z0:
                        has_higher = True
                        break
                if has_higher:
                    high_edges.append((i, j))

            # Low edge: (1) has defined flow (i.e., not NOFLOW and not NODATA_FD)
            #          (2) has a same-elevation neighbor
            #          (3) that neighbor is NOFLOW
            if fd0 != NOFLOW and fd0 != NODATA_FD:
                has_same_noflow = False
                for di, dj in D8_OFFS:
                    ni, nj = i + di, j + dj
                    if not _inb(ni, nj, nrows, ncols) or not valid[ni, nj]:
                        continue
                    if Zf[ni, nj] == z0 and flow[ni, nj] == NOFLOW:
                        has_same_noflow = True
                        break
                if has_same_noflow:
                    low_edges.append((i, j))

    if len(high_edges) == 0 and len(low_edges) == 0:
        # no flats
        dem_out = Zf.copy()
        if np.isnan(nodata):
            dem_out[~valid] = np.nan
        else:
            dem_out[~valid] = nodata
        labels = np.zeros((nrows, ncols), dtype=np.int32)
        flatmask = np.zeros((nrows, ncols), dtype=np.int32)
        stats = {"n_flats": 0, "n_flat_cells": 0}
        return dem_out, flatmask, labels, stats

    # If there are high edges but no low edges => undrainable flats (paper says: stop)
    if len(low_edges) == 0 and len(high_edges) > 0:
        dem_out = Zf.copy()
        if np.isnan(nodata):
            dem_out[~valid] = np.nan
        else:
            dem_out[~valid] = nodata
        labels = np.zeros((nrows, ncols), dtype=np.int32)
        flatmask = np.zeros((nrows, ncols), dtype=np.int32)
        stats = {"n_flats": 0, "n_flat_cells": 0, "undrainable_flats": True}
        return dem_out, flatmask, labels, stats

    # ---- Step 1b: label drainable flats via flood-fill from LowEdges ----
    labels = np.zeros((nrows, ncols), dtype=np.int32)
    label_id = 0

    # We seed from LowEdges cells, and flood-fill across equal-elevation cells,
    # but only across cells which are part of flat (NOFLOW) or low-edge structure.
    # In practice, the flood-fill region is "the flat surface at elevation e".
    for seed_i, seed_j in list(low_edges):
        if labels[seed_i, seed_j] != 0:
            continue

        label_id += 1
        e = Zf[seed_i, seed_j]
        q = deque([(seed_i, seed_j)])
        labels[seed_i, seed_j] = label_id

        while q:
            ci, cj = q.popleft()
            for di, dj in D8_OFFS:
                ni, nj = ci + di, cj + dj
                if not _inb(ni, nj, nrows, ncols) or not valid[ni, nj]:
                    continue
                if labels[ni, nj] != 0:
                    continue
                if Zf[ni, nj] != e:
                    continue
                # Only cells on the same flat level are part of the component
                labels[ni, nj] = label_id
                q.append((ni, nj))

    # Remove HighEdges that are not labeled => undrainable flats get excluded
    filtered_high = deque()
    while high_edges:
        i, j = high_edges.popleft()
        if labels[i, j] != 0:
            filtered_high.append((i, j))
    high_edges = filtered_high

    # If nothing remains labeled, nothing to do
    if label_id == 0:
        dem_out = Zf.copy()
        if np.isnan(nodata):
            dem_out[~valid] = np.nan
        else:
            dem_out[~valid] = nodata
        flatmask = np.zeros((nrows, ncols), dtype=np.int32)
        stats = {"n_flats": 0, "n_flat_cells": 0}
        return dem_out, flatmask, labels, stats

    # ---- Step 2: BFS away-from-higher (queue with iteration marker) ----
    flatmask = np.zeros((nrows, ncols), dtype=np.int32)
    flatheights = np.zeros(label_id + 1, dtype=np.int32)

    marker = (-1, -1)
    q = deque(high_edges)
    q.append(marker)
    I = 1

    while len(q) > 1:
        ci, cj = q.popleft()
        if (ci, cj) == marker:
            I += 1
            q.append(marker)
            continue

        # Only process unlabeled? No: only labeled flats
        lbl = labels[ci, cj]
        if lbl == 0:
            continue

        # Only increment each cell once
        if flatmask[ci, cj] != 0:
            continue

        flatmask[ci, cj] = I
        if I > flatheights[lbl]:
            flatheights[lbl] = I

        # Expand only into same-labeled cells which have NOFLOW
        for di, dj in D8_OFFS:
            ni, nj = ci + di, cj + dj
            if not _inb(ni, nj, nrows, ncols) or not valid[ni, nj]:
                continue
            if labels[ni, nj] != lbl:
                continue
            if flow[ni, nj] != NOFLOW:
                continue
            if flatmask[ni, nj] == 0:
                q.append((ni, nj))

    # ---- Step 3: BFS towards-lower + combine (double weight to lower-gradient) ----
    # Make all flatmask entries for labeled flats negative (as in the paper's description)
    flatmask[labels > 0] *= -1

    q = deque(low_edges)
    q.append(marker)
    I = 1

    while len(q) > 1:
        ci, cj = q.popleft()
        if (ci, cj) == marker:
            I += 1
            q.append(marker)
            continue

        lbl = labels[ci, cj]
        if lbl == 0:
            continue

        # Add neighbors first (same label, NOFLOW)
        for di, dj in D8_OFFS:
            ni, nj = ci + di, cj + dj
            if not _inb(ni, nj, nrows, ncols) or not valid[ni, nj]:
                continue
            if labels[ni, nj] != lbl:
                continue
            if flow[ni, nj] != NOFLOW:
                continue
            if flatmask[ni, nj] < 0:  # still unprocessed in step 3
                q.append((ni, nj))

        # Combine gradients:
        # - If cell had a step-2 value (negative), invert using FlatHeights and add 2I.
        # - Otherwise just set to 2I.
        if flatmask[ci, cj] < 0:
            # flatmask is negative: -(step2 distance)
            step2 = -flatmask[ci, cj]
            flatmask[ci, cj] = (flatheights[lbl] - step2) + 2 * I
        else:
            flatmask[ci, cj] = 2 * I

    # ---- Output DEM (optionally) ----
    dem_out = Zf.copy()
    if apply_to_dem:
        sel = (labels > 0) & valid & (flatmask > 0)
        dem_out[sel] = dem_out[sel] + epsilon * flatmask[sel]

    if np.isnan(nodata):
        dem_out[~valid] = np.nan
    else:
        dem_out[~valid] = nodata

    stats = {
        "n_flats": int(label_id),
        "n_flat_cells": int(np.count_nonzero(labels)),
        "boundary_flows_out": bool(boundary_flows_out),
        "apply_to_dem": bool(apply_to_dem),
        "epsilon": float(epsilon),
    }
    return dem_out, flatmask.astype(np.int32), labels, stats

# ---------------------------------------------------
# 8-neighborhood offsets (D8)
OFFS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def resolve_flats_garbrecht_martz_1997(
    dem: np.ndarray,
    nodata: float = np.nan,
    *,
    vertical_resolution: float = 1.0,
    equal_tol: float = 0.0,
    lower_tol: float = 0.0,
    treat_oob_as_lower: bool = True,
    max_exception_iters: int = 50,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, float]]:
    """
    Garbrecht & Martz (1997) flat-resolution algorithm.

    Implements:
      Step 1: Gradient towards lower terrain (backward growth from outlets).
      Step 2: Gradient away from higher terrain (growth from higher edges; previously
              incremented cells are incremented again each pass -> produces a reversed distance ramp).
      Step 3: Linear addition of both gradients, apply to DEM using an infinitesimal increment.
      Exceptional situation: If increments cancel, apply half-increment following Step 1 repeatedly
                             until no flat cell remains.

    Parameters
    ----------
    dem : np.ndarray
        2D DEM.
    nodata : float
        NoData marker (np.nan supported).
    vertical_resolution : float
        Vertical DEM resolution vr (meters etc.). Paper uses increment = 2/100000 * vr.
    equal_tol : float
        Tolerance for treating elevations as equal when defining a flat surface.
        Set 0.0 for strict (closest to the paper).
    lower_tol : float
        Tolerance for strictly lower comparisons; dz < -lower_tol means strictly lower.
        Set 0.0 for strict.
    treat_oob_as_lower : bool
        Paper assumes boundary cells can drain outward. This models out-of-bounds as lower.
    max_exception_iters : int
        Safety cap for exceptional-case iterations.

    Returns
    -------
    dem_out : np.ndarray
        DEM modified by tiny increments (float64).
    fields : dict[str, np.ndarray]
        Diagnostic fields:
          - labels: int32 flat-surface labels (0 = not in any flat surface)
          - inc_towards: int32 increments from Step 1
          - inc_away: int32 increments from Step 2
          - inc_total: int32 total increments (Step 3, without half-fixes)
          - inc_half_added: int32 extra "half increments" applied in exceptional fix
    stats : dict[str, float]
        Summary stats and used increments.
    """
    Z = np.asarray(dem, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError("DEM must be a 2D array.")

    nrows, ncols = Z.shape

    # Valid-data mask
    if np.isnan(nodata):
        valid = np.isfinite(Z)
    else:
        valid = (Z != nodata) & np.isfinite(Z)

    def inb(r: int, c: int) -> bool:
        return 0 <= r < nrows and 0 <= c < ncols

    # --- Helper: strict-equality within tolerance ---
    def is_equal(a: float, b: float) -> bool:
        return abs(a - b) <= equal_tol

    # --- Helper: determine if a cell has a strictly lower neighbor (in a given surface) ---
    def has_strict_lower_neighbor(surface: np.ndarray, r: int, c: int, base_z: float) -> bool:
        for dr, dc in OFFS8:
            nr, nc = r + dr, c + dc
            if not inb(nr, nc) or (not valid[nr, nc]):
                if treat_oob_as_lower:
                    return True
                continue
            if surface[nr, nc] < base_z - lower_tol:
                return True
        return False

    # --- 1) Identify flat surfaces as connected components of equal elevation
    # A "flat surface" here means: a connected region of (approximately) equal elevation
    # that contains at least one cell without a strictly-lower neighbor AND has at least one outlet
    # (a cell in the region adjacent to strictly-lower terrain).
    labels = np.zeros((nrows, ncols), dtype=np.int32)
    label_id = 0

    # Precompute candidate cells: any valid cell; we will build plateaus by equality components.
    visited = np.zeros((nrows, ncols), dtype=bool)

    # These arrays will store per-label masks: outlet seeds and high-edge seeds
    # (we keep lists of coordinates to avoid huge per-label boolean stacks).
    outlet_seeds: Dict[int, list[tuple[int, int]]] = {}
    highedge_seeds: Dict[int, list[tuple[int, int]]] = {}
    flat_cells_count = 0
    drainable_flats_count = 0

    for r in range(nrows):
        for c in range(ncols):
            if not valid[r, c] or visited[r, c]:
                continue

            # Flood-fill connected component of equal elevation (within equal_tol).
            base_z = Z[r, c]
            q = deque([(r, c)])
            visited[r, c] = True
            comp = [(r, c)]

            while q:
                cr, cc = q.popleft()
                for dr, dc in OFFS8:
                    nr, nc = cr + dr, cc + dc
                    if not inb(nr, nc) or (not valid[nr, nc]) or visited[nr, nc]:
                        continue
                    if is_equal(Z[nr, nc], base_z):
                        visited[nr, nc] = True
                        q.append((nr, nc))
                        comp.append((nr, nc))

            # Decide if this component is a "flat surface" requiring treatment:
            # - contains at least one cell with no strictly-lower neighbor
            # - contains at least one outlet cell (adjacent to strictly-lower terrain)
            any_noflow = False
            any_outlet = False

            # For Step 1, seeds are cells in the component that ARE adjacent to lower terrain.
            comp_outlets: list[tuple[int, int]] = []

            # For Step 2, initial high edges are cells adjacent to higher terrain AND not adjacent to lower terrain.
            comp_highedges: list[tuple[int, int]] = []

            for (cr, cc) in comp:
                # Outlet check: adjacent to strictly lower than base_z
                is_outlet = False
                is_adj_higher = False

                for dr, dc in OFFS8:
                    nr, nc = cr + dr, cc + dc
                    if not inb(nr, nc) or (not valid[nr, nc]):
                        if treat_oob_as_lower:
                            is_outlet = True
                        continue
                    dz = Z[nr, nc] - base_z
                    if dz < -lower_tol:
                        is_outlet = True
                    elif dz > equal_tol:
                        is_adj_higher = True

                if is_outlet:
                    any_outlet = True
                    comp_outlets.append((cr, cc))
                else:
                    # If no strictly-lower neighbor, the cell has no local downslope exit within the DEM
                    any_noflow = True

                # High-edge (Step 2) seed definition from paper:
                # adjacent to higher terrain AND no adjacent lower terrain.
                if (not is_outlet) and is_adj_higher:
                    comp_highedges.append((cr, cc))

            # If it is not "flat" (no noflow cells), skip
            # If it has no outlet, it cannot drain (G&M requires at least one lower neighbor at edge).
            if (not any_noflow) or (not any_outlet):
                continue

            # Label the component as a flat surface
            label_id += 1
            for (cr, cc) in comp:
                labels[cr, cc] = label_id
            outlet_seeds[label_id] = comp_outlets
            highedge_seeds[label_id] = comp_highedges
            flat_cells_count += len(comp)
            drainable_flats_count += 1

    # If no flats, return early
    if label_id == 0:
        out = Z.copy()
        out[~valid] = (np.nan if np.isnan(nodata) else nodata)
        fields = {
            "labels": labels,
            "inc_towards": np.zeros_like(labels, dtype=np.int32),
            "inc_away": np.zeros_like(labels, dtype=np.int32),
            "inc_total": np.zeros_like(labels, dtype=np.int32),
            "inc_half_added": np.zeros_like(labels, dtype=np.int32),
        }
        stats = {
            "n_flats": 0.0,
            "n_flat_cells": 0.0,
            "increment_unit": 2.0 * vertical_resolution / 100000.0,
            "half_increment_unit": vertical_resolution / 100000.0,
        }
        return out, fields, stats

    # Paper's increment units
    inc_unit = 2.0 * vertical_resolution / 100000.0
    half_unit = 1.0 * vertical_resolution / 100000.0

    # --- Step 1: Gradient towards lower terrain (distance from outlets) ---
    inc_towards = np.zeros((nrows, ncols), dtype=np.int32)

    for lbl in range(1, label_id + 1):
        # BFS within this label from outlet seeds; outlets get 0 increments.
        seeds = outlet_seeds.get(lbl, [])
        if not seeds:
            # Not drainable; should not occur due to filtering.
            continue

        dist = np.full((nrows, ncols), -1, dtype=np.int32)
        q = deque()

        for (sr, sc) in seeds:
            dist[sr, sc] = 0
            q.append((sr, sc))

        while q:
            cr, cc = q.popleft()
            d0 = dist[cr, cc]
            for dr, dc in OFFS8:
                nr, nc = cr + dr, cc + dc
                if not inb(nr, nc):
                    continue
                if labels[nr, nc] != lbl:
                    continue
                if dist[nr, nc] != -1:
                    continue
                dist[nr, nc] = d0 + 1
                q.append((nr, nc))

        # Cells at distance d>0 are incremented by d (Step 1 "passes")
        sel = (labels == lbl) & (dist > 0)
        inc_towards[sel] = dist[sel]

    # --- Step 2: Gradient away from higher terrain ---
    # This step produces a REVERSED distance ramp:
    # cells closest to higher terrain get the largest increment counts.
    inc_away = np.zeros((nrows, ncols), dtype=np.int32)

    for lbl in range(1, label_id + 1):
        # Eligible cells: flat cells NOT adjacent to lower terrain (paper excludes outlet-adjacent cells here).
        # We reconstruct "adjacent to lower" using the Step 1 seeds list.
        outlet_set = set(outlet_seeds.get(lbl, []))

        eligible = np.zeros((nrows, ncols), dtype=bool)
        region_idx = np.argwhere(labels == lbl)
        for (cr, cc) in region_idx:
            eligible[cr, cc] = (cr, cc) not in outlet_set

        # Initial high-edge seeds for Step 2 (may be empty in some geometries).
        seeds = highedge_seeds.get(lbl, [])

        if not seeds:
            # If no high edges exist, Step 2 contributes nothing for this flat surface.
            continue

        dist = np.full((nrows, ncols), -1, dtype=np.int32)
        q = deque()

        for (sr, sc) in seeds:
            if not eligible[sr, sc]:
                continue
            dist[sr, sc] = 0
            q.append((sr, sc))

        while q:
            cr, cc = q.popleft()
            d0 = dist[cr, cc]
            for dr, dc in OFFS8:
                nr, nc = cr + dr, cc + dc
                if not inb(nr, nc):
                    continue
                if not eligible[nr, nc]:
                    continue
                if dist[nr, nc] != -1:
                    continue
                dist[nr, nc] = d0 + 1
                q.append((nr, nc))

        # Reverse the ramp to match "previously incremented cells are incremented again each pass"
        # Desired final increments:
        #   high-edge cells (dist=0) get max+1 increments,
        #   next ring gets max, ... farthest gets 1.
        max_d = dist[eligible].max() if np.any(dist[eligible] >= 0) else -1
        if max_d < 0:
            continue

        sel = eligible & (dist >= 0)
        inc_away[sel] = (max_d + 1) - dist[sel]

    # --- Step 3: Combine increments and apply to DEM ---
    inc_total = inc_towards + inc_away
    dem_out = Z.copy()
    apply = (labels > 0) & valid & (inc_total > 0)
    dem_out[apply] = dem_out[apply] + inc_unit * inc_total[apply]

    # --- Exceptional situation: cancellation creates new noflow cells ---
    # We detect remaining noflow cells inside each labeled flat, then apply Step 1 again
    # using HALF increment unit until resolved.
    inc_half_added = np.zeros((nrows, ncols), dtype=np.int32)

    def compute_noflow_mask(surface: np.ndarray, lbl: int, base_z: float) -> np.ndarray:
        """Cells in label lbl that have no strictly-lower neighbor in 'surface'."""
        m = np.zeros((nrows, ncols), dtype=bool)
        coords = np.argwhere(labels == lbl)
        for (cr, cc) in coords:
            z0 = surface[cr, cc]
            # In a perfect G&M scenario, these should be very close to base_z + increments.
            # We use z0 for correctness after increments.
            has_lower = False
            for dr, dc in OFFS8:
                nr, nc = cr + dr, cc + dc
                if not inb(nr, nc) or (not valid[nr, nc]):
                    if treat_oob_as_lower:
                        has_lower = True
                    continue
                if surface[nr, nc] < z0 - lower_tol:
                    has_lower = True
            if not has_lower:
                m[cr, cc] = True
        return m

    exception_iters_used = 0

    for lbl in range(1, label_id + 1):
        # Safety: skip if somehow not drainable
        if not outlet_seeds.get(lbl):
            continue

        # Iterate half-increment fix until no noflow remains or cap reached.
        for _ in range(max_exception_iters):
            # Detect unresolved cells (no downslope neighbor) within this flat after Step 3.
            # If none, done.
            noflow = compute_noflow_mask(dem_out, lbl, 0.0)
            if not np.any(noflow):
                break

            # Build BFS distance from cells that DO have a downslope (i.e., NOT in noflow)
            # This matches "repeat Step 1" behavior: increment cells lacking drainage until they drain.
            region = (labels == lbl)
            seeds = np.argwhere(region & (~noflow))

            if seeds.size == 0:
                # No seed with drainage exists => cannot fix by Step 1 (should not happen for drainable flats)
                break

            dist = np.full((nrows, ncols), -1, dtype=np.int32)
            q = deque()

            for (sr, sc) in seeds:
                dist[sr, sc] = 0
                q.append((sr, sc))

            while q:
                cr, cc = q.popleft()
                d0 = dist[cr, cc]
                for dr, dc in OFFS8:
                    nr, nc = cr + dr, cc + dc
                    if not inb(nr, nc):
                        continue
                    if labels[nr, nc] != lbl:
                        continue
                    if dist[nr, nc] != -1:
                        continue
                    dist[nr, nc] = d0 + 1
                    q.append((nr, nc))

            # Apply half increments only where needed (noflow cells), proportionally to distance.
            # One "pass" of Step 1 corresponds to +1 increment, hence +half_unit here.
            sel = noflow & (dist > 0)
            if not np.any(sel):
                # If only seeds are noflow, fallback: bump all noflow by 1 half increment.
                sel = noflow

            dem_out[sel] = dem_out[sel] + half_unit * 1.0
            inc_half_added[sel] += 1

            exception_iters_used += 1

        # end per-label fix loop

    # Restore NoData
    if np.isnan(nodata):
        dem_out[~valid] = np.nan
    else:
        dem_out[~valid] = nodata

    fields = {
        "labels": labels.astype(np.int32),
        "inc_towards": inc_towards.astype(np.int32),
        "inc_away": inc_away.astype(np.int32),
        "inc_total": inc_total.astype(np.int32),
        "inc_half_added": inc_half_added.astype(np.int32),
    }

    stats = {
        "n_flats": float(label_id),
        "n_flat_cells": float(flat_cells_count),
        "n_flats_drainable": float(drainable_flats_count),
        "increment_unit": float(inc_unit),
        "half_increment_unit": float(half_unit),
        "exception_iters_used": float(exception_iters_used),
        "equal_tol": float(equal_tol),
        "lower_tol": float(lower_tol),
        "vertical_resolution": float(vertical_resolution),
    }

    return dem_out, fields, stats


def resolve_flats_towards_lower_edge(
    dem,
    nodata=np.nan,
    *,
    epsilon=2e-5,             # velikost kroků (neovlivní počet změn, jen Δ)
    equal_tol=1e-3,           # tolerance pro "rovnost" (spojování plošin / ties)
    lower_tol=0.0,            # tolerance pro "přísně nižší"
    equality_connectivity=8,  # 4 nebo 8 pro spojování plošin; PySheds je blíž 8
    treat_oob_as_lower=True,  # hrana rastru jako nižší terén (běžná praxe)
    cascade_equal_to_lower=True,  # "rovný soused" mimo plošinu, který SÁM má nižšího -> považuj za lower-edge
    force_all_flats=False,    # když plošina nemá lower-edge, použij perimetr (ne zcela hydrologické)
    bump_frontier=True        # posuň i frontier (dist==0), zvýší počet změn jako u PySheds
):
    """
    Garbrecht–Martz styl: vynucení minimálního sklonu TOWARDS nižšímu okraji
    + "kaskádové" výtoky přes rovné sousedy (ties). Vše plně na numpy, BFS.

    Návrat:
      dem_out  : float32 DEM po úpravě (NoData zachováno)
      flatmask : int32   vzdálenost od použité frontier uvnitř plošiny (0..)
      labels   : int32   ID plošin (0 = mimo plochu)
      stats    : dict    počty a parametry běhu
    """
    Z = np.asarray(dem)
    if Z.ndim != 2:
        raise ValueError("DEM must be 2D")
    nrows, ncols = Z.shape

    # valid mask
    if np.isnan(nodata):
        valid = np.isfinite(Z)
    else:
        valid = (Z != nodata) & np.isfinite(Z)

    OFFS8 = [(-1,-1), (-1,0), (-1,1),
             ( 0,-1),         ( 0,1),
             ( 1,-1), ( 1,0), ( 1,1)]
    OFFS4 = [(-1,0), (1,0), (0,-1), (0,1)]
    OFFS_EQ = OFFS4 if equality_connectivity == 4 else OFFS8

    def inb(i,j): return (0 <= i < nrows) and (0 <= j < ncols)

    # ---- 1) Kdo má přísně nižšího souseda? (8-conn, lower_tol) ----
    has_lower = np.zeros_like(valid, bool)
    for di, dj in OFFS8:
        i0, i1 = max(0, -di), min(nrows, nrows - di)
        j0, j1 = max(0, -dj), min(ncols, ncols - dj)
        if i0 >= i1 or j0 >= j1:
            continue
        a = Z[i0:i1, j0:j1]
        b = Z[i0+di:i1+di, j0+dj:j1+dj]
        v = valid[i0:i1, j0:j1] & valid[i0+di:i1+di, j0+dj:j1+dj]
        has_lower[i0:i1, j0:j1] |= v & ((b - a) < -lower_tol)

    # Plošiny = valid & NEMÁ nižšího souseda (rovnost se zde nevyžaduje)
    flats_mask = valid & (~has_lower)

    # ---- 2) Labeling plošin (rovnost v toleranci, 4/8-conn) ----
    labels = np.zeros_like(valid, np.int32)
    cur = 0
    for i in range(nrows):
        for j in range(ncols):
            if flats_mask[i, j] and labels[i, j] == 0:
                cur += 1
                z0 = Z[i, j]
                q = deque([(i, j)])
                labels[i, j] = cur
                while q:
                    ci, cj = q.popleft()
                    for di, dj in OFFS_EQ:
                        ni, nj = ci + di, cj + dj
                        if inb(ni, nj) and flats_mask[ni, nj] and labels[ni, nj] == 0:
                            if abs(Z[ni, nj] - z0) <= equal_tol:
                                labels[ni, nj] = cur
                                q.append((ni, nj))

    if cur == 0:
        out = Z.astype(np.float32, copy=True)
        out[~valid] = (np.nan if np.isnan(nodata) else nodata)
        stats = {"n_flats": 0, "n_flat_cells": 0, "n_changed_cells": 0}
        return out, np.zeros_like(labels), labels, stats

    # ---- 3) Pro každou plošinu určete frontier: skutečný lower-edge + "kaskáda" ----
    def bfs_distance(region_mask, frontier_mask):
        """8-conn BFS vzdálenost od frontier; -1 = nedosaženo."""
        dist = np.full(region_mask.shape, -1, np.int32)
        q = deque()
        fi, fj = np.where(frontier_mask & region_mask)
        for a, b in zip(fi, fj):
            dist[a, b] = 0
            q.append((a, b))
        while q:
            ci, cj = q.popleft()
            d0 = dist[ci, cj]
            for di, dj in OFFS8:
                ni, nj = ci + di, cj + dj
                if inb(ni, nj) and region_mask[ni, nj] and dist[ni, nj] == -1:
                    dist[ni, nj] = d0 + 1
                    q.append((ni, nj))
        return dist

    dem_out = Z.astype(np.float32, copy=True)
    flatmask = np.zeros_like(Z, np.int32)
    n_changed = 0
    n_drainable = 0
    n_forced = 0

    for lbl in range(1, cur + 1):
        region = (labels == lbl)
        if not np.any(region):
            continue
        flat_z = Z[region][0]

        # a) skutečný lower-edge + "cascaded ties"
        lower_edge = np.zeros_like(region, bool)
        ri, rj = np.where(region)
        for ci, cj in zip(ri, rj):
            for di, dj in OFFS8:
                ni, nj = ci + di, cj + dj
                if not inb(ni, nj) or not valid[ni, nj]:
                    if treat_oob_as_lower:
                        lower_edge[ci, cj] = True
                    continue
                if not region[ni, nj]:
                    dz = Z[ni, nj] - flat_z
                    if dz < -lower_tol:
                        lower_edge[ci, cj] = True
                    elif cascade_equal_to_lower and abs(dz) <= equal_tol:
                        # soused mimo plošinu je "rovný"; ověř kaskádově, že on sám má nižšího
                        got_lower = False
                        for ddi, ddj in OFFS8:
                            mi, mj = ni + ddi, nj + ddj
                            if inb(mi, mj) and valid[mi, mj]:
                                if Z[mi, mj] < Z[ni, nj] - lower_tol:
                                    got_lower = True
                                    break
                        if got_lower:
                            lower_edge[ci, cj] = True

        use_fallback = False
        if np.any(lower_edge):
            frontier = lower_edge
            n_drainable += 1
        else:
            if not force_all_flats:
                continue
            # fallback: perimetr plošiny
            perim = np.zeros_like(region, bool)
            for ci, cj in zip(ri, rj):
                for di, dj in OFFS8:
                    ni, nj = ci + di, cj + dj
                    if not inb(ni, nj) or not region[ni, nj]:
                        perim[ci, cj] = True
                        break
            frontier = perim
            use_fallback = True
            n_forced += 1

        # b) BFS vzdálenost dovnitř plošiny
        dist = bfs_distance(region, frontier)

        sel = region & (dist >= 0)
        if not np.any(sel):
            continue

        flatmask[sel] = dist[sel]

        if bump_frontier:
            eff = dist[sel] + 1  # frontier (0) dostane krok 1
            dem_out[sel] = dem_out[sel] + (epsilon * eff).astype(np.float32)
            n_changed += int(sel.sum())
        else:
            core = sel & (dist > 0)
            dem_out[core] = dem_out[core] + (epsilon * dist[core]).astype(np.float32)
            n_changed += int(np.count_nonzero(core))

    # NoData zpět
    if np.isnan(nodata):
        dem_out[~valid] = np.nan
    else:
        dem_out[~valid] = nodata

    stats = {
        "n_flats": int(cur),
        "n_flat_cells": int(np.count_nonzero(labels)),
        "n_flats_drainable": int(n_drainable),
        "n_flats_forced": int(n_forced),
        "n_changed_cells": int(n_changed),
        "equal_tol": float(equal_tol),
        "lower_tol": float(lower_tol),
        "equality_connectivity": int(equality_connectivity),
        "treat_oob_as_lower": bool(treat_oob_as_lower),
        "cascade_equal_to_lower": bool(cascade_equal_to_lower),
        "force_all_flats": bool(force_all_flats),
        "bump_frontier": bool(bump_frontier),
        "epsilon": float(epsilon),
    }
    return dem_out, flatmask.astype(np.int32), labels, stats

#----------------------------------------------------

import numpy as np
from collections import deque

def resolve_flats_towards_lower_edge_gm(
    dem,
    nodata=np.nan,
    epsilon=2e-5,
    equal_tol=1e-3,           # rovnost pro labelování plošin
    lower_tol=0.0,            # prah pro "přísně nižšího" souseda
    equality_connectivity=4,  # 4 je bližší literatuře; 8 lze povolit
    treat_oob_as_lower=True,  # na ořezu často pomůže True
    force_all_flats=False,    # řešit i bez skutečného výtoku (fallback = perimetr)
    bump_frontier=False,      # zvýšit i frontier (dist_low==0)
    cascade_equal_to_lower=True  # „tie“: rovný soused sám má nižšího -> brát jako lower-edge
):
    Z = np.asarray(dem)
    if Z.ndim != 2:
        raise ValueError("DEM must be 2D")
    nrows, ncols = Z.shape

    valid = np.isfinite(Z) if np.isnan(nodata) else ((Z != nodata) & np.isfinite(Z))

    OFFS8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    OFFS4 = [(-1,0),(1,0),(0,-1),(0,1)]
    OFFS_EQ = OFFS4 if equality_connectivity==4 else OFFS8

    def inb(i,j): return 0 <= i < nrows and 0 <= j < ncols

    # --- 1) má buňka přísně nižšího souseda? (8-conn, s lower_tol)
    has_lower = np.zeros_like(valid, bool)
    for di,dj in OFFS8:
        i0,i1 = max(0,-di), min(nrows, nrows-di)
        j0,j1 = max(0,-dj), min(ncols, ncols-dj)
        if i0>=i1 or j0>=j1: 
            continue
        a = Z[i0:i1, j0:j1]
        b = Z[i0+di:i1+di, j0+dj:j1+dj]
        v = valid[i0:i1, j0:j1] & valid[i0+di:i1+di, j0+dj:j1+dj]
        has_lower[i0:i1, j0:j1] |= v & ((b - a) < -lower_tol)

    # plošiny = buňky bez přísně nižšího souseda
    flats_mask = valid & (~has_lower)

    # --- 2) labelování plošin (rovnost ±equal_tol; 4/8-conn pro rovnost)
    labels = np.full(Z.shape, 0, np.int32)
    cur = 0
    for i in range(nrows):
        for j in range(ncols):
            if flats_mask[i,j] and labels[i,j]==0:
                cur += 1
                z0 = Z[i,j]
                q = deque([(i,j)])
                labels[i,j] = cur
                while q:
                    ci,cj = q.popleft()
                    for di,dj in OFFS_EQ:
                        ni,nj = ci+di, cj+dj
                        if (inb(ni,nj) and flats_mask[ni,nj] and labels[ni,nj]==0
                            and abs(Z[ni,nj]-z0) <= equal_tol):
                            labels[ni,nj] = cur
                            q.append((ni,nj))

    if cur == 0:
        out = Z.astype(np.float32, copy=True)
        out[~valid] = (np.nan if np.isnan(nodata) else nodata)
        return out, np.zeros_like(Z, np.int32), labels, {
            "n_flats":0,"n_flat_cells":0,"n_changed_cells":0,"n_drainable":0
        }

    # --- 3) vybuduj low-edge & high-edge pro každou plošinu
    LowEdge  = [deque() for _ in range(cur+1)]
    HighEdge = [deque() for _ in range(cur+1)]
    has_low  = np.zeros(cur+1, bool)

    for i in range(nrows):
        for j in range(ncols):
            lbl = labels[i,j]
            if lbl==0: continue
            z0 = Z[i,j]
            adj_low   = False
            adj_high  = False
            is_oob = False
            for di,dj in OFFS8:
                ni,nj = i+di, j+dj
                if not inb(ni,nj) or not valid[ni,nj]:
                    is_oob = True
                    if treat_oob_as_lower:
                        adj_low = True
                    continue
                if labels[ni,nj] != lbl:
                    dz = Z[ni,nj] - z0
                    if dz < -lower_tol:
                        adj_low = True
                    elif abs(dz) <= equal_tol and cascade_equal_to_lower and has_lower[ni,nj]:
                        adj_low = True
                    elif dz >  equal_tol:
                        adj_high = True
            if adj_low:
                LowEdge[lbl].append((i,j)); has_low[lbl] = True
            if adj_high:
                HighEdge[lbl].append((i,j))

    # --- 4) dvě BFS uvnitř plošiny: D_high (od vyššího), D_low (k nižšímu)
    D_high = np.full(Z.shape, -1, np.int32)
    D_low  = np.full(Z.shape, -1, np.int32)

    # od vyššího
    for lbl in range(1, cur+1):
        if len(HighEdge[lbl])==0: 
            continue
        q = deque(HighEdge[lbl])
        for si,sj in q:
            D_high[si,sj] = 1  # mark-on-enqueue
        while q:
            ci,cj = q.popleft()
            d = D_high[ci,cj]
            for di,dj in OFFS8:
                ni,nj = ci+di, cj+dj
                if inb(ni,nj) and labels[ni,nj]==lbl and D_high[ni,nj]==-1:
                    D_high[ni,nj] = d + 1
                    q.append((ni,nj))

    # k nižšímu
    drainable = 0
    for lbl in range(1, cur+1):
        region_has_low = has_low[lbl]
        if not region_has_low and not force_all_flats:
            continue
        # seedy
        seeds = LowEdge[lbl]
        if (not region_has_low) and force_all_flats:
            # fallback: perimetr plošiny
            seeds = deque()
            ii,jj = np.where(labels==lbl)
            region = (labels==lbl)
            for ci,cj in zip(ii,jj):
                for di,dj in OFFS8:
                    ni,nj = ci+di, cj+dj
                    if not inb(ni,nj) or not region[ni,nj]:
                        seeds.append((ci,cj)); break
        else:
            drainable += 1

        if len(seeds)==0:
            continue

        q = deque(seeds)
        for si,sj in q:
            D_low[si,sj] = 0  # frontier distance = 0
        while q:
            ci,cj = q.popleft()
            d = D_low[ci,cj]
            for di,dj in OFFS8:
                ni,nj = ci+di, cj+dj
                if inb(ni,nj) and labels[ni,nj]==lbl and D_low[ni,nj]==-1:
                    D_low[ni,nj] = d + 1
                    q.append((ni,nj))

    # --- 5) kombinace (Garbrecht & Martz): G = 2*D_low + (H - D_high)
    FlatMask = np.zeros_like(Z, np.int32)
    for lbl in range(1, cur+1):
        region = (labels==lbl)
        if not np.any(region): 
            continue
        # H = max po plošině (jen kde D_high>=0)
        H = 0
        if np.any((D_high>=0) & region):
            H = int(D_high[(D_high>=0) & region].max())
        # frontier bump?
        inc_low = D_low.copy()
        if bump_frontier:
            inc_low[(D_low==0) & region] = 1
        # kombinace
        g = np.zeros_like(FlatMask)
        # jen buňky plošiny, kam se dostal D_low (tj. region vyřešený)
        sel = region & (D_low>=0)
        if np.any(sel):
            g[sel] = 2*inc_low[sel]
            # kde máme i D_high, přičti (H - D_high)
            both = sel & (D_high>=0)
            if np.any(both):
                g[both] += (H - D_high[both])
            FlatMask[sel] = g[sel]

    # --- 6) aplikace Δ jen uvnitř plošin
    out = Z.astype(np.float32, copy=True)
    inc = (labels>0) & valid & (FlatMask!=0)
    out[inc] = out[inc] + (epsilon * FlatMask[inc]).astype(np.float32)
    out[~valid] = (np.nan if np.isnan(nodata) else nodata)

    stats = {
        "n_flats": int(cur),
        "n_flat_cells": int(np.count_nonzero(labels)),
        "n_changed_cells": int(np.count_nonzero(inc)),
        "n_flats_drainable": int(drainable),
        "equal_tol": float(equal_tol),
        "lower_tol": float(lower_tol),
        "equality_connectivity": int(equality_connectivity),
        "treat_oob_as_lower": bool(treat_oob_as_lower),
        "force_all_flats": bool(force_all_flats),
        "bump_frontier": bool(bump_frontier),
        "epsilon": float(epsilon),
    }
    return out, FlatMask.astype(np.int32), labels, stats

# -------------------------------------------------------
import numpy as np
from collections import deque

# nejlip funguje
def resolve_flats_barnes(
    dem,
    nodata=np.nan,
    epsilon=2e-5,
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

# asi lepsi
# import numpy as np
# from collections import deque
# from typing import Tuple, Dict

# def resolve_flats_barnes(
#     dem: np.ndarray,
#     nodata: float = np.nan,
#     *,
#     epsilon: float = 2e-5,
#     equal_tol: float = 3e-3,         # default tuned for 30 m FABDEM & PySheds parity
#     lower_tol: float = 0.0,          # strict "lower" detection
#     treat_oob_as_lower: bool = True,
#     require_low_edge_only: bool = True,
#     force_all_flats: bool = False
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
#     """
#     Resolve flat areas in a filled DEM by superimposing a tiny monotone gradient
#     away from higher edges and toward lower edges (Barnes 2014, improved GM'97).
#     Only flats with outlets are modified when require_low_edge_only=True.

#     Parameters
#     ----------
#     dem : 2D array (float)
#         Filled DEM (no depressions). Must be 2D.
#     nodata : float or NaN
#         NoData marker. NaNs are treated as invalid.
#     epsilon : float
#         Step size for the synthetic gradient.
#     equal_tol : float
#         Equality tolerance for plateau membership & tie checks.
#     lower_tol : float
#         Strictness for "neighbor is lower" tests (dz < -lower_tol).
#     treat_oob_as_lower : bool
#         Treat raster boundary as a lower edge (recommended True).
#     require_low_edge_only : bool
#         If True, resolve only flats that have an actual outlet (Barnes).
#     force_all_flats : bool
#         If True and a plateau has no outlet, force resolution by seeding its perimeter.

#     Returns
#     -------
#     dem_out : float32 DEM with tiny gradient imposed over resolved flats
#     flatmask : int32 field of weights added (unitless)
#     labels : int32 plateau labels (0 = non-flat)
#     stats : dict with counts
#     """
#     Z = np.asarray(dem)
#     if Z.ndim != 2:
#         raise ValueError("DEM must be 2D")
#     nrows, ncols = Z.shape

#     # Valid mask
#     if np.isnan(nodata):
#         valid = np.isfinite(Z)
#     else:
#         valid = (Z != nodata) & np.isfinite(Z)

#     OFFS8 = [(-1,-1),(-1,0),(-1,1),
#              ( 0,-1),       ( 0,1),
#              ( 1,-1),( 1,0),( 1,1)]
#     def inb(i, j): return (0 <= i < nrows) and (0 <= j < ncols)

#     # --- 1) Precompute: has a strictly lower neighbor? (with lower_tol)
#     has_lower8 = np.zeros_like(valid, dtype=bool)
#     for di, dj in OFFS8:
#         i0, i1 = max(0, -di), min(nrows, nrows - di)
#         j0, j1 = max(0, -dj), min(ncols, ncols - dj)
#         if i0 >= i1 or j0 >= j1:
#             continue
#         a = Z[i0:i1, j0:j1]
#         b = Z[i0+di:i1+di, j0+dj:j1+dj]
#         v = valid[i0:i1, j0:j1] & valid[i0+di:i1+di, j0+dj:j1+dj]
#         # neighbor lower if (b - a) < -lower_tol  <=>  a - b > lower_tol
#         has_lower8[i0:i1, j0:j1] |= v & ((b - a) < -lower_tol)

#     # Flat cells: valid & no strictly lower neighbor
#     flats = valid & (~has_lower8)

#     # --- 2) Label plateaus (8-connected), compare to CURRENT cell (not seed)
#     labels = np.zeros_like(valid, dtype=np.int32)
#     cur = 0
#     for i in range(nrows):
#         for j in range(ncols):
#             if flats[i, j] and labels[i, j] == 0:
#                 cur += 1
#                 q = deque([(i, j)])
#                 labels[i, j] = cur
#                 while q:
#                     ci, cj = q.popleft()
#                     zc = Z[ci, cj]
#                     for di, dj in OFFS8:
#                         ni, nj = ci + di, cj + dj
#                         if (inb(ni, nj)
#                             and flats[ni, nj]
#                             and labels[ni, nj] == 0
#                             and abs(Z[ni, nj] - zc) <= equal_tol):
#                             labels[ni, nj] = cur
#                             q.append((ni, nj))

#     if cur == 0:
#         dem_out = Z.astype(np.float32, copy=True)
#         dem_out[~valid] = (np.nan if np.isnan(nodata) else nodata)
#         stats = {"n_flats": 0, "n_flats_active": 0, "n_flat_cells": 0, "n_changed_cells": 0}
#         return dem_out, np.zeros_like(labels, np.int32), labels, stats

#     # --- 3) Build edge queues & outlet flags per plateau
#     HighEdges = [deque() for _ in range(cur + 1)]
#     LowEdges  = [deque() for _ in range(cur + 1)]
#     has_low   = np.zeros(cur + 1, dtype=bool)
#     has_high  = np.zeros(cur + 1, dtype=bool)

#     for i in range(nrows):
#         for j in range(ncols):
#             lbl = labels[i, j]
#             if lbl == 0:
#                 continue
#             z0 = Z[i, j]
#             adj_higher = False
#             adj_lower  = False
#             is_boundary = False
#             for di, dj in OFFS8:
#                 ni, nj = i + di, j + dj
#                 if not inb(ni, nj) or not valid[ni, nj]:
#                     is_boundary = True
#                     if treat_oob_as_lower:
#                         adj_lower = True
#                     continue
#                 if labels[ni, nj] != lbl:
#                     dz = Z[ni, nj] - z0
#                     if dz >  equal_tol:
#                         adj_higher = True
#                     # strictly lower neighbor
#                     if dz < -lower_tol:
#                         adj_lower = True
#                     # cascade: equal neighbor which itself has a lower neighbor
#                     elif abs(dz) <= equal_tol and has_lower8[ni, nj]:
#                         adj_lower = True

#             if adj_lower:
#                 LowEdges[lbl].append((i, j)); has_low[lbl]  = True
#             if adj_higher:
#                 HighEdges[lbl].append((i, j)); has_high[lbl] = True

#             # Optional fallback: if forcing closed plateaus, seed perimeter as pseudo-low
#             if force_all_flats and not adj_lower:
#                 if (not inb(i-1, j)) or (not inb(i+1, j)) or (not inb(i, j-1)) or (not inb(i, j+1)):
#                     # already boundary; treated above
#                     pass

#     # If forcing flats without outlets, seed full perimeter now
#     if force_all_flats:
#         for lbl in range(1, cur + 1):
#             if has_low[lbl]:
#                 continue
#             # perimeter = at least one neighbor not in the label (or invalid/OOB)
#             for i in range(nrows):
#                 for j in range(ncols):
#                     if labels[i, j] != lbl:
#                         continue
#                     perim = False
#                     for di, dj in OFFS8:
#                         ni, nj = i + di, j + dj
#                         if not inb(ni, nj) or (not valid[ni, nj]) or (labels[ni, nj] != lbl):
#                             perim = True
#                             break
#                     if perim:
#                         LowEdges[lbl].append((i, j))
#             if len(LowEdges[lbl]) > 0:
#                 has_low[lbl] = True

#     def flat_active(lbl: int) -> bool:
#         if require_low_edge_only:
#             return has_low[lbl]       # resolve only flats with outlets (Barnes)
#         # otherwise: allow either natural outlet or forced perimeter
#         return has_low[lbl] or force_all_flats

#     # --- 4) Two BFS passes: away-from-higher, toward-lower; combine weights
#     away     = np.full(labels.shape, -1, dtype=np.int32)
#     towards  = np.full(labels.shape, -1, dtype=np.int32)
#     FlatMask = np.zeros_like(labels, dtype=np.int32)
#     FlatH    = np.zeros(cur + 1, dtype=np.int32)

#     # A) From higher edge (away)
#     for lbl in range(1, cur + 1):
#         if not flat_active(lbl):
#             continue
#         q = HighEdges[lbl]
#         if not q:
#             continue
#         for si, sj in q:
#             away[si, sj] = 1
#         while q:
#             ci, cj = q.popleft()
#             cl = away[ci, cj]
#             if cl > FlatH[lbl]:
#                 FlatH[lbl] = cl
#             for di, dj in OFFS8:
#                 ni, nj = ci + di, cj + dj
#                 if inb(ni, nj) and labels[ni, nj] == lbl and away[ni, nj] == -1:
#                     away[ni, nj] = cl + 1
#                     q.append((ni, nj))

#     # B) Toward lower edge (dominant 2*dist) and combine with away
#     drainable = 0
#     active_flats = 0
#     for lbl in range(1, cur + 1):
#         if not flat_active(lbl):
#             continue
#         q = LowEdges[lbl]
#         if not q:
#             continue
#         active_flats += 1
#         if has_low[lbl]:
#             drainable += 1

#         for si, sj in q:
#             towards[si, sj] = 1
#         while q:
#             ci, cj = q.popleft()
#             cl = towards[ci, cj]
#             if away[ci, cj] != -1:
#                 FlatMask[ci, cj] = (FlatH[lbl] - away[ci, cj]) + 2 * cl
#             else:
#                 FlatMask[ci, cj] = 2 * cl
#             for di, dj in OFFS8:
#                 ni, nj = ci + di, cj + dj
#                 if inb(ni, nj) and labels[ni, nj] == lbl and towards[ni, nj] == -1:
#                     towards[ni, nj] = cl + 1
#                     q.append((ni, nj))

#     # --- 5) Apply tiny gradient over active flats
#     dem_out = Z.astype(np.float32, copy=True)
#     inc = (labels > 0) & valid & (FlatMask != 0)
#     dem_out[inc] = dem_out[inc] + epsilon * FlatMask[inc]

#     # keep nodata
#     if np.isnan(nodata):
#         dem_out[~valid] = np.nan
#     else:
#         dem_out[~valid] = nodata

#     stats = {
#         "n_flats": int(cur),
#         "n_flats_active": int(active_flats),
#         "n_flats_drainable": int(drainable),
#         "n_flat_cells": int(np.count_nonzero(labels)),
#         "n_changed_cells": int(np.count_nonzero(inc)),
#         "equal_tol": float(equal_tol),
#         "lower_tol": float(lower_tol),
#         "require_low_edge_only": bool(require_low_edge_only),
#         "force_all_flats": bool(force_all_flats),
#     }
#     return dem_out, FlatMask.astype(np.int32), labels, stats
#-------------------------------------------------------------------
# TIE MODIFIKACE- dava zmenenych 10 k ale...
import numpy as np
from collections import deque

def resolve_flats_barnes_tie_slow(
    dem,
    nodata=np.nan,
    epsilon=2e-5,           # sníženo o řád (kvůli rozsahu Δ)
    equal_tol=1e-3,
    lower_tol=0.0,
    treat_oob_as_lower=True,
    require_low_edge_only=True,
    force_all_flats=False,
    include_equal_ties=True  # <<< NOVÉ: zapnout „tie“ buňky (jako pysheds)
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

    # --- "má nižšího souseda?" (8-conn, s prahem lower_tol) ---
    has_lower8 = np.zeros_like(valid, bool)
    for di,dj in OFFS8:
        i0,i1 = max(0,-di), min(nrows, nrows-di)
        j0,j1 = max(0,-dj), min(ncols, ncols-dj)
        if i0>=i1 or j0>=j1: continue
        a = Z[i0:i1, j0:j1]
        b = Z[i0+di:i1+di, j0+dj:j1+dj]
        v = valid[i0:i1, j0:j1] & valid[i0+di:i1+di, j0+dj:j1+dj]
        has_lower8[i0:i1, j0:j1] |= v & ((b - a) < -lower_tol)

    # kandidáti na "flats" (bez přísně nižšího souseda)
    flats = valid & (~has_lower8)

    # --- labelování (8-conn) ---
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
                        if (inb(ni,nj) and flats[ni,nj] and labels[ni,nj]==0
                            and abs(Z[ni,nj]-z0) <= equal_tol):
                            labels[ni,nj] = cur
                            q.append((ni,nj))

    # --- AUGMENTACE: připojit "tie" buňky (rovné ± equal_tol), i když nejsou flats ---
    if include_equal_ties and cur > 0:
        for lbl in range(1, cur+1):
            region = (labels == lbl)
            if not np.any(region): 
                continue
            z0 = Z[region][0]
            q = deque(list(zip(*np.where(region))))
            while q:
                ci,cj = q.popleft()
                for di,dj in OFFS8:
                    ni,nj = ci+di, cj+dj
                    if inb(ni,nj) and valid[ni,nj] and labels[ni,nj]==0:
                        if abs(Z[ni,nj]-z0) <= equal_tol:
                            labels[ni,nj] = lbl
                            q.append((ni,nj))

    if cur == 0:
        dem_out = Z.astype(np.float32, copy=True)
        dem_out[~valid] = (np.nan if np.isnan(nodata) else nodata)
        stats = {"n_flats":0,"n_flat_cells":0,"n_changed_cells":0,"n_flats_drainable":0}
        return dem_out, np.zeros_like(labels), labels, stats

    # --- hrany (kaskáda: rovný soused, který sám má nižšího) ---
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
                    if dz < -lower_tol: adj_lower  = True
                    elif abs(dz) <= equal_tol and has_lower8[ni,nj]:
                        adj_lower = True
            if adj_lower:
                LowEdges[lbl].append((i,j)); has_low[lbl]=True
            if adj_higher:
                HighEdges[lbl].append((i,j)); has_high[lbl]=True
            if force_all_flats and is_boundary and not adj_lower:
                LowEdges[lbl].append((i,j)); has_low[lbl]=True

    def flat_active(lbl):
        return has_low[lbl] or force_all_flats

    # --- dvě BFS, mark-on-enqueue ---
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

    # B) k nižšímu + kombinace
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

    # --- aplikace Δ ---
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
        "include_equal_ties": bool(include_equal_ties),
        "epsilon": float(epsilon),
    }
    return dem_out, FlatMask.astype(np.int32), labels, stats

# TIE FAST ----------------------------------------
import numpy as np
from collections import deque
from typing import Tuple, Dict

OFFS8 = [(-1,-1),(-1,0),(-1,1),
         ( 0,-1),       ( 0,1),
         ( 1,-1),( 1,0),( 1,1)]

def resolve_flats_barnes_tie(
    dem: np.ndarray,
    nodata: float = np.nan,
    *,
    epsilon: float = 2e-5,          # tiny monotone gradient step
    equal_tol: float = 3e-3,        # tuned for 30 m FABDEM (parity with PySheds)
    lower_tol: float = 0.0,         # strict "lower" test
    treat_oob_as_lower: bool = True,
    require_low_edge_only: bool = True,
    force_all_flats: bool = False,
    include_equal_ties: bool = True # include equal-elevation "ties" into plateau during labeling
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Barnes/GM-style flat resolution with a single-pass labeling that can optionally
    absorb equal-elevation 'tie' cells during BFS expansion. Complexity ~O(N).

    Returns:
        dem_out: float32 DEM with epsilon*FlatMask added on flat areas
        FlatMask: int32 weights (superposition of 'away' and 'towards' BFS)
        labels: int32 plateau labels (0 = non-flat)
        stats: dict with counts
    """
    Z = np.asarray(dem, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError("DEM must be 2D")
    nrows, ncols = Z.shape

    # Valid mask
    valid = np.isfinite(Z) if np.isnan(nodata) else ((Z != nodata) & np.isfinite(Z))

    def inb(i, j): return (0 <= i < nrows) and (0 <= j < ncols)

    # 1) Precompute: has strictly-lower neighbor?  (vectorized windowing per offset)
    has_lower8 = np.zeros_like(valid, dtype=bool)
    for di, dj in OFFS8:
        i0, i1 = max(0, -di), min(nrows, nrows - di)
        j0, j1 = max(0, -dj), min(ncols, ncols - dj)
        if i0 >= i1 or j0 >= j1:
            continue
        a = Z[i0:i1, j0:j1]
        b = Z[i0+di:i1+di, j0+dj:j1+dj]
        v = valid[i0:i1, j0:j1] & valid[i0+di:i1+di, j0+dj:j1+dj]
        has_lower8[i0:i1, j0:j1] |= v & ((b - a) < -lower_tol)

    # Flats = valid & no strictly-lower neighbor
    flats = valid & (~has_lower8)

    # 2) Single-pass plateau labeling:
    #    - Seeds are true flats.
    #    - Expansion criterion:
    #        a) always 8-connected
    #        b) |Z[nb] - Z[cur]| <= equal_tol
    #        c) if include_equal_ties: neighbor may be non-flat; else must be flats[nb]
    labels = np.zeros_like(valid, dtype=np.int32)
    cur = 0
    for i in range(nrows):
        for j in range(ncols):
            if flats[i, j] and labels[i, j] == 0:
                cur += 1
                q = deque([(i, j)])
                labels[i, j] = cur
                while q:
                    ci, cj = q.popleft()
                    zc = Z[ci, cj]
                    for di, dj in OFFS8:
                        ni, nj = ci + di, cj + dj
                        if not inb(ni, nj) or not valid[ni, nj] or labels[ni, nj] != 0:
                            continue
                        if abs(Z[ni, nj] - zc) > equal_tol:
                            continue
                        if (not include_equal_ties) and (not flats[ni, nj]):
                            continue
                        labels[ni, nj] = cur
                        q.append((ni, nj))

    if cur == 0:
        dem_out = Z.copy() 
        dem_out[~valid] = (np.nan if np.isnan(nodata) else nodata)
        stats = {"n_flats": 0, "n_flats_active": 0, "n_flats_drainable": 0,
                 "n_flat_cells": 0, "n_changed_cells": 0}
        return dem_out, np.zeros_like(labels, np.int32), labels, stats

    # 3) Build edges per plateau (cascade to lower through equal neighbors that have lower)
    HighEdges = [deque() for _ in range(cur + 1)]
    LowEdges  = [deque() for _ in range(cur + 1)]
    has_low   = np.zeros(cur + 1, dtype=bool)

    for i in range(nrows):
        for j in range(ncols):
            lbl = labels[i, j]
            if lbl == 0:
                continue
            z0 = Z[i, j]
            adj_higher = False
            adj_lower  = False
            is_boundary = False
            for di, dj in OFFS8:
                ni, nj = i + di, j + dj
                if not inb(ni, nj) or not valid[ni, nj]:
                    is_boundary = True
                    if treat_oob_as_lower:
                        adj_lower = True
                    continue
                if labels[ni, nj] != lbl:
                    dz = Z[ni, nj] - z0
                    if dz >  equal_tol:
                        adj_higher = True
                    if dz < -lower_tol:
                        adj_lower = True
                    elif abs(dz) <= equal_tol and has_lower8[ni, nj]:
                        # cascade: equal neighbor which itself has a lower neighbor
                        adj_lower = True
            if adj_lower:
                LowEdges[lbl].append((i, j)); has_low[lbl] = True
            if adj_higher:
                HighEdges[lbl].append((i, j))

            # optional fallback for closed plateaus
            if force_all_flats and not adj_lower and is_boundary:
                LowEdges[lbl].append((i, j)); has_low[lbl] = True

    # If forcing closed plateaus: seed their full perimeter once (cheap single pass)
    if force_all_flats:
        for lbl in range(1, cur + 1):
            if has_low[lbl]:
                continue
            # perimeter = at least one neighbor not in the label (or invalid/OOB)
            seeded = False
            for i in range(nrows):
                for j in range(ncols):
                    if labels[i, j] != lbl:
                        continue
                    for di, dj in OFFS8:
                        ni, nj = i + di, j + dj
                        if not inb(ni, nj) or (not valid[ni, nj]) or (labels[ni, nj] != lbl):
                            LowEdges[lbl].append((i, j))
                            seeded = True
                            break
                # micro-early exit if we already found some perimeter cells
            if seeded:
                has_low[lbl] = True

    def flat_active(lbl: int) -> bool:
        if require_low_edge_only:
            return has_low[lbl]         # resolve only flats with outlets
        return has_low[lbl] or force_all_flats

    # 4) Two BFS passes: away-from-higher & towards-lower; combine as FlatMask
    away     = np.full(labels.shape, -1, dtype=np.int32)
    towards  = np.full(labels.shape, -1, dtype=np.int32)
    FlatMask = np.zeros_like(labels, dtype=np.int32)
    FlatH    = np.zeros(cur + 1, dtype=np.int32)

    # A) away-from-higher
    for lbl in range(1, cur + 1):
        if not flat_active(lbl):
            continue
        q = HighEdges[lbl]
        if not q:
            continue
        for si, sj in q:
            away[si, sj] = 1
        while q:
            ci, cj = q.popleft()
            cl = away[ci, cj]
            if cl > FlatH[lbl]:
                FlatH[lbl] = cl
            for di, dj in OFFS8:
                ni, nj = ci + di, cj + dj
                if 0 <= ni < nrows and 0 <= nj < ncols and labels[ni, nj] == lbl and away[ni, nj] == -1:
                    away[ni, nj] = cl + 1
                    q.append((ni, nj))

    # B) towards-lower (dominant) + combine
    drainable = 0
    active_flats = 0
    for lbl in range(1, cur + 1):
        if not flat_active(lbl):
            continue
        q = LowEdges[lbl]
        if not q:
            continue
        active_flats += 1
        if has_low[lbl]:
            drainable += 1
        for si, sj in q:
            towards[si, sj] = 1
        while q:
            ci, cj = q.popleft()
            cl = towards[ci, cj]
            if away[ci, cj] != -1:
                FlatMask[ci, cj] = (FlatH[lbl] - away[ci, cj]) + 2 * cl
            else:
                FlatMask[ci, cj] = 2 * cl
            for di, dj in OFFS8:
                ni, nj = ci + di, cj + dj
                if 0 <= ni < nrows and 0 <= nj < ncols and labels[ni, nj] == lbl and towards[ni, nj] == -1:
                    towards[ni, nj] = cl + 1
                    q.append((ni, nj))

    # 5) Apply epsilon*FlatMask
    dem_out = Z.copy()
    inc = (labels > 0) & valid & (FlatMask != 0)
    dem_out[inc] = dem_out[inc] + epsilon * FlatMask[inc]

    if np.isnan(nodata):
        dem_out[~valid] = np.nan
    else:
        dem_out[~valid] = nodata

    stats = {
        "n_flats": int(cur),
        "n_flats_active": int(active_flats),
        "n_flats_drainable": int(drainable),
        "n_flat_cells": int(np.count_nonzero(labels)),
        "n_changed_cells": int(np.count_nonzero(inc)),
        "equal_tol": float(equal_tol),
        "lower_tol": float(lower_tol),
        "require_low_edge_only": bool(require_low_edge_only),
        "force_all_flats": bool(force_all_flats),
        "include_equal_ties": bool(include_equal_ties),
        "epsilon": float(epsilon),
    }
    return dem_out, FlatMask.astype(np.int32), labels, stats

# -------------------------------------------------------
# Cascade
def resolve_flats_barnes_cascade(
    dem,
    nodata=np.nan,
    epsilon=2e-5,
    equal_tol=1e-6,           # přísné spojování plošin
    lower_tol=0.0,            # „nižší“ je opravdu nižší
    treat_oob_as_lower=False, # <<< vypnuto, ať to neteče ven z domény
    require_low_edge_only=True,
    force_all_flats=False,
    cascade_equal_to_lower=False  # <<< nový přepínač, default vypnuto
):
    Z = np.asarray(dem)
    nrows, ncols = Z.shape
    valid = np.isfinite(Z) if np.isnan(nodata) else ((Z != nodata) & np.isfinite(Z))

    OFFS8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    def inb(i,j): return 0 <= i < nrows and 0 <= j < ncols

    # 1) buňky bez přísně nižšího souseda
    has_lower8 = np.zeros_like(valid, bool)
    for di,dj in OFFS8:
        i0,i1 = max(0,-di), min(nrows, nrows-di)
        j0,j1 = max(0,-dj), min(ncols, ncols-dj)
        if i0>=i1 or j0>=j1: continue
        a = Z[i0:i1, j0:j1]
        b = Z[i0+di:i1+di, j0+dj:j1+dj]
        v = valid[i0:i1, j0:j1] & valid[i0+di:i1+di, j0+dj:j1+dj]
        has_lower8[i0:i1, j0:j1] |= v & ((b - a) < -lower_tol)

    flats = valid & (~has_lower8)

    # 2) labelování plošin (jen přísně rovné v toleranci equal_tol)
    labels = np.zeros_like(valid, np.int32)
    cur = 0
    from collections import deque
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
        return dem_out, np.zeros_like(labels), labels, {"n_flats":0,"n_flat_cells":0,"n_changed_cells":0}

    # 3) hrany
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
                    if dz < -lower_tol: adj_lower  = True
                    elif cascade_equal_to_lower and (abs(dz) <= equal_tol) and has_lower8[ni,nj]:
                        adj_lower = True
            if adj_lower:  LowEdges[lbl].append((i,j)); has_low[lbl]=True
            if adj_higher: HighEdges[lbl].append((i,j)); has_high[lbl]=True
            if force_all_flats and is_boundary and not adj_lower:
                LowEdges[lbl].append((i,j)); has_low[lbl] = True

    def flat_active(lbl):
        if has_low[lbl]: return True
        return force_all_flats

    # 4) dvě BFS
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
                if 0 <= ni < nrows and 0 <= nj < ncols and labels[ni,nj]==lbl and away[ni,nj]==-1:
                    away[ni,nj] = cl + 1
                    q.append((ni,nj))

    # B) k nižšímu (dominantní 2*dist)
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
                if 0 <= ni < nrows and 0 <= nj < ncols and labels[ni,nj]==lbl and towards[ni,nj]==-1:
                    towards[ni,nj] = cl + 1
                    q.append((ni,nj))

    # 5) aplikace pouze uvnitř plošin
    dem_out = Z.astype(np.float32, copy=True)
    inc = (labels>0) & valid
    dem_out[inc] = dem_out[inc] + epsilon * FlatMask[inc]
    dem_out[~valid] = (np.nan if np.isnan(nodata) else nodata)

    stats = {
        "n_flats": int(cur),
        "n_flats_drainable": int(drainable),
        "n_flat_cells": int(np.count_nonzero(labels)),
        "n_changed_cells": int(np.count_nonzero((FlatMask!=0) & inc)),
        "equal_tol": float(equal_tol),
        "lower_tol": float(lower_tol),
        "cascade_equal_to_lower": bool(cascade_equal_to_lower),
        "treat_oob_as_lower": bool(treat_oob_as_lower),
    }
    return dem_out, FlatMask.astype(np.int32), labels, stats

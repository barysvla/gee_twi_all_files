from __future__ import annotations

import numpy as np


def meters_per_degree_lat_lon(lat_deg: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Compute meters per 1 degree of latitude/longitude at given latitude(s) using WGS84."""
    # WGS84 ellipsoid parameters
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = (2.0 - f) * f

    # Vectorized latitude (radians)
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    s = np.sin(lat)
    c = np.cos(lat)

    # Radii of curvature (meridional M and prime vertical N)
    one_minus_e2s2 = 1.0 - e2 * s * s
    M = a * (1.0 - e2) / (one_minus_e2s2 ** 1.5)
    N = a / np.sqrt(one_minus_e2s2)

    # Convert degrees to meters along meridians/parallels
    m_per_deg_lat = (np.pi / 180.0) * M
    m_per_deg_lon = (np.pi / 180.0) * N * np.clip(c, 0.0, None)
    return m_per_deg_lat, m_per_deg_lon


def step_lengths_for_rows_epsg4326(transform, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-row distances (meters) between pixel centers for a north-up EPSG:4326 grid.
    """
    # Pixel size in degrees (absolute values; north-up rasters typically have negative transform.e)
    deg_x = float(abs(transform.a))
    deg_y = float(abs(transform.e))

    # transform.f is the Y origin (latitude at the top edge), transform.e is the row step (negative)
    lat_origin = float(transform.f)
    lat_step = float(transform.e)

    # Latitude at pixel centers for each row (vector length = height)
    lat_centers_deg = lat_origin + lat_step * (np.arange(height, dtype=np.float64) + 0.5)

    # Meters per degree at each row latitude
    mlat, mlon = meters_per_degree_lat_lon(lat_centers_deg)

    # Convert degree pixel sizes to meters (row-dependent for longitude; latitude is ~constant-ish but computed exactly)
    dx = mlon * deg_x
    dy = mlat * deg_y
    d_diag = np.hypot(dx, dy)
    return dx.astype(np.float64), dy.astype(np.float64), d_diag.astype(np.float64)


# D8 neighbor offsets in fixed order: NE, E, SE, S, SW, W, NW, N
# This fixed ordering is useful to keep consistency with MFD/FD8-style implementations.
D8_OFFSETS = np.array(
    [
        (-1,  1),  # NE
        ( 0,  1),  # E
        ( 1,  1),  # SE
        ( 1,  0),  # S
        ( 1, -1),  # SW
        ( 0, -1),  # W
        (-1, -1),  # NW
        (-1,  0),  # N
    ],
    dtype=np.int32,
)


def compute_flow_direction_d8(
    dem: np.ndarray,
    transform,
    *,
    nodata_mask: np.ndarray | None = None,
    min_slope: float = 0.0,
    out_dtype=np.int16,
) -> np.ndarray:
    """
    Compute D8 flow directions on a hydrologically corrected DEM.

    For each cell, the downslope neighbor with maximum
        tan(beta) = (z_center - z_neighbor) / d
    is selected in an 8-neighborhood.

    Directions are encoded as integers:
        0..7 = [NE, E, SE, S, SW, W, NW, N]
        -1   = NoData.

    Assumes EPSG:4326 input (degrees): distances d are computed in meters per row,
    and the DEM has no flats or pits.
    """
    # --- Input normalization -------------------------------------------------
    # Use float64 internally for stable slope comparisons.
    z = np.asarray(dem, dtype=np.float64)
    h, w = z.shape

    # --- NoData mask ---------------------------------------------------------
    # If user did not provide a mask, treat non-finite cells (NaN/Inf) as NoData.
    if nodata_mask is None:
        nodata = ~np.isfinite(z)
    else:
        nodata = np.asarray(nodata_mask, dtype=bool)

    # --- Row-wise metric distances (EPSG:4326) -------------------------------
    # In EPSG:4326, cell sizes are in degrees; the metric distance of 1 degree of longitude
    # depends on latitude. We precompute dx, dy, and diagonal distance for each row.
    dx_row, dy_row, d_diag_row = step_lengths_for_rows_epsg4326(transform, h)

    # --- Output allocation ---------------------------------------------------
    # Direction index per cell:
    #   -1 for NoData (and for "no downslope neighbor", which should not occur for corrected DEMs)
    #    0..7 for the chosen D8 neighbor direction in the fixed order.
    dir_idx = np.full((h, w), -1, dtype=out_dtype)

    # --- Main scan over raster cells -----------------------------------------
    # Two nested loops: row-major traversal. This is straightforward and matches typical
    # DEM processing patterns; it is also easy to port to numba/cython later if needed.
    for i in range(h):
        # Precompute the 8 neighbor distances for this row in the same order as D8_OFFSETS.
        # Cardinal directions use dx or dy; diagonal directions use hypot(dx, dy).
        step_len = np.array(
            [d_diag_row[i], dx_row[i], d_diag_row[i], dy_row[i],
             d_diag_row[i], dx_row[i], d_diag_row[i], dy_row[i]],
            dtype=np.float64,
        )

        for j in range(w):
            # Skip NoData cells: direction stays -1.
            if nodata[i, j]:
                continue

            zc = z[i, j]

            # Track the best (steepest) downslope neighbor.
            # We maximize tan(beta) = dz / d.
            best_k = -1
            best_tan = -np.inf

            # --- Evaluate 8 neighbors ---------------------------------------
            # For each direction k, compute:
            #   dz = z_center - z_neighbor
            #   tan(beta) = dz / distance
            # and keep the neighbor with the maximum tan(beta) among downslope neighbors.
            for k in range(8):
                di, dj = int(D8_OFFSETS[k, 0]), int(D8_OFFSETS[k, 1])
                ni, nj = i + di, j + dj

                # Bounds check: neighbors outside raster are ignored.
                if not (0 <= ni < h and 0 <= nj < w):
                    continue

                # Ignore NoData neighbors (treat as absent).
                if nodata[ni, nj]:
                    continue

                # Downslope requirement: only strictly lower neighbors are eligible.
                dz = zc - z[ni, nj]
                if dz <= 0.0:
                    continue

                # Metric distance between pixel centers for this neighbor direction.
                d = float(step_len[k])
                if d <= 0.0:
                    continue

                # Candidate slope (tangent of slope angle).
                tan_beta = dz / d

                # Optional suppression of tiny slopes (numerical noise control).
                if tan_beta <= float(min_slope):
                    continue

                # Update best direction if this neighbor is steeper.
                # For a corrected DEM, ties are not expected; no explicit tie-break is used.
                if tan_beta > best_tan:
                    best_tan = tan_beta
                    best_k = k

            # Store the selected direction index.
            # For corrected DEMs, best_k should be != -1 for all valid cells.
            dir_idx[i, j] = best_k if best_k != -1 else -1

    # Ensure NoData cells are consistently encoded as -1.
    dir_idx[nodata] = -1
    return dir_idx

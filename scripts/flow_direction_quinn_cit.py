import numpy as np

def compute_flow_direction_quinn_cit(dem, scale=1.0, cit=1000.0, h=2.0):
    """
    Compute flow direction weights using Quinn 1995 with CIT (Channel Initiation Threshold).

    Parameters:
        dem (2D np.ndarray): elevation model
        scale (float): resolution (cell size in meters)
        cit (float): channel initiation threshold (in contributing area units)
        h (float): exponent controlling p sensitivity to a (typically 1–2)

    Returns:
        flow_weights (3D np.ndarray): flow proportions to 8 neighbors in D8 order
    """
    rows, cols = dem.shape
    flow = np.zeros((rows, cols, 8), dtype=np.float32)

    D8 = [
        (-1, 1),  # NE
        (0, 1),   # E
        (1, 1),   # SE
        (1, 0),   # S
        (1, -1),  # SW
        (0, -1),  # W
        (-1, -1), # NW
        (-1, 0)   # N
    ]
    distances = np.array([np.sqrt(2), 1, np.sqrt(2), 1, np.sqrt(2), 1, np.sqrt(2), 1]) * scale
    L = distances.copy()  # Assume L_i = d_i for simplification (can be modified)

    # Initial contributing area per cell (in pixel units)
    a = np.ones((rows, cols), dtype=np.float32)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            center = dem[i, j]
            slopes = []
            valid_indices = []

            for k, (dy, dx) in enumerate(D8):
                ni, nj = i + dy, j + dx
                dz = center - dem[ni, nj]
                tan_beta = dz / distances[k] if dz > 0 else 0
                if tan_beta > 0:
                    slopes.append((L[k] * tan_beta))
                    valid_indices.append(k)

            if slopes:
                p = ((a[i, j] / cit) + 1) ** h
                powered = [s**p for s in slopes]
                total = sum(powered)
                if total > 0:
                    for idx, k in enumerate(valid_indices):
                        flow[i, j, k] = powered[idx] / total

    return flow

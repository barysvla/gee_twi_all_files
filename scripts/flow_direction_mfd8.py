import numpy as np

def compute_flow_direction_mfd8(dem, p=1.1):
    """
    Compute MFD8 flow directions for a given DEM using Freeman's approach.

    Parameters:
        dem (2D np.ndarray): input elevation model
        p (float): exponent that controls convergence/divergence of flow (typically 1.1 - 2.0)

    Returns:
        directions (3D np.ndarray): array of shape (rows, cols, 8), where each cell contains
                                    proportion of flow to its 8 neighbors in D8 order.
    """
    rows, cols = dem.shape
    flow = np.zeros((rows, cols, 8), dtype=np.float32)

    # Define D8 directions: (dy, dx)
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

    distances = np.array([np.sqrt(2), 1, np.sqrt(2), 1, np.sqrt(2), 1, np.sqrt(2), 1])

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            center = dem[i, j]
            slopes = []

            for k, (dy, dx) in enumerate(D8):
                ni, nj = i + dy, j + dx
                dz = center - dem[ni, nj]
                slope = dz / distances[k] if dz > 0 else 0
                slopes.append(slope ** p)

            total = sum(slopes)
            if total > 0:
                flow[i, j, :] = np.array(slopes) / total

    return flow

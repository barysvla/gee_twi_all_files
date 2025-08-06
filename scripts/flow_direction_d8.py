import numpy as np
import matplotlib.pyplot as plt

# D8 directional offsets and codes
D8_OFFSETS = [
    (-1, 1),   # NE = 128
    (0, 1),    # E  = 1
    (1, 1),    # SE = 2
    (1, 0),    # S  = 4
    (1, -1),   # SW = 8
    (0, -1),   # W  = 16
    (-1, -1),  # NW = 32
    (-1, 0)    # N  = 64
]
D8_CODES = [128, 1, 2, 4, 8, 16, 32, 64]
D8_DIST = [np.sqrt(2), 1, np.sqrt(2), 1, np.sqrt(2), 1, np.sqrt(2), 1]

def compute_d8_direction(dem: np.ndarray) -> np.ndarray:
    """
    Compute D8 flow direction from a DEM array.
    Each cell flows to the neighbor with the steepest downslope.

    Parameters:
        dem (2D np.ndarray): Input DEM as a NumPy array.

    Returns:
        flow_dir (2D np.ndarray): D8 direction codes for each cell.
    """
    rows, cols = dem.shape
    flow_dir = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            max_slope = -np.inf
            direction = 0
            center = dem[i, j]

            # Check all 8 neighbors
            for k, (dy, dx) in enumerate(D8_OFFSETS):
                ni, nj = i + dy, j + dx
                neighbor = dem[ni, nj]
                dz = center - neighbor
                slope = dz / D8_DIST[k]

                # Save the steepest downslope direction
                if slope > max_slope and slope > 0:
                    max_slope = slope
                    direction = D8_CODES[k]

            flow_dir[i, j] = direction

    return flow_dir

def visualize_d8_direction(direction: np.ndarray):
    """
    Visualize D8 flow direction codes using matplotlib.

    Parameters:
        direction (2D np.ndarray): Output from compute_d8_direction.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(direction, cmap="twilight", interpolation="none")
    ax.set_title("D8 Flow Direction")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

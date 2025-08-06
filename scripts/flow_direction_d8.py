import numpy as np

# D8 direction codes and neighbor offsets
D8_DIRECTIONS = [
    (-1, 1),  # 128 NE
    (0, 1),   # 1   E
    (1, 1),   # 2   SE
    (1, 0),   # 4   S
    (1, -1),  # 8   SW
    (0, -1),  # 16  W
    (-1, -1), # 32  NW
    (-1, 0)   # 64  N
]

D8_CODES = [128, 1, 2, 4, 8, 16, 32, 64]
MIN_DROP_THRESHOLD = 1e-4  # Minimum drop to be considered valid

def compute_flow_direction_d8(dem):
    """
    Compute D8 flow direction for each cell in DEM.
    Each cell is assigned a direction code indicating the direction
    of steepest descent toward one of its 8 neighbors.

    Parameters:
        dem (ndarray): 2D numpy array representing the elevation model.

    Returns:
        ndarray: 2D array of same shape with D8 direction codes (uint8).
    """
    direction = np.zeros_like(dem, dtype=np.uint8)
    rows, cols = dem.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            center = dem[i, j]
            max_drop = -np.inf
            best_idx = -1

            for idx, (di, dj) in enumerate(D8_DIRECTIONS):
                ni, nj = i + di, j + dj
                neighbor = dem[ni, nj]
                distance = np.sqrt(di**2 + dj**2)  # Euclidean distance
                drop = (center - neighbor) / distance

                if drop > max_drop and drop > MIN_DROP_THRESHOLD:
                    max_drop = drop
                    best_idx = idx

            if best_idx >= 0:
                best_code = D8_CODES[best_idx]
            else:
                best_code = 0  # No valid flow direction

            direction[i, j] = best_code

    return direction


def visualize_d8_direction(direction):
    """
    Visualize D8 flow direction codes as an image.

    Parameters:
        direction (ndarray): 2D array of direction codes.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(direction, cmap="twilight", interpolation="none")
    ax.set_title("D8 Flow Direction")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

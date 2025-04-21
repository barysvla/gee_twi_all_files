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

def visualize_mfd8_direction(flow_array, direction_names=None):
    """
    Visualize the MFD8 flow proportions in 8 directions as subplots.

    Parameters:
        flow_array (ndarray): 3D array of shape (rows, cols, 8)
        direction_names (list of str): optional labels for directions
    """
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 4, figsize=(16, 6))
    axs = axs.flatten()
    
    if direction_names is None:
        direction_names = ["NE", "E", "SE", "S", "SW", "W", "NW", "N"]

    for i in range(8):
        axs[i].imshow(flow_array[:, :, i], cmap="Blues")
        axs[i].set_title(f"Flow to {direction_names[i]}")
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_mfd8_combined(flow_array):
    """
    Visualize combined MFD8 direction as a single image using dominant direction.
    Parameters:
        flow_array (ndarray): 3D array of shape (rows, cols, 8)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.patches as mpatches

    direction_labels = ["NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    dominant = np.argmax(flow_array, axis=2)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(dominant, cmap="tab10", interpolation="none")
    ax.set_title("MFD8 Dominant Flow Direction (by max weight)")

    # Legend with direction names
    handles = [mpatches.Patch(color=plt.cm.tab10(i), label=direction_labels[i]) for i in range(8)]
    fig.colorbar(im, ax=ax, ticks=range(8), fraction=0.046, pad=0.04)
    ax.legend(handles=handles, title="Direction", loc="lower right", bbox_to_anchor=(1.4, 0))

    plt.tight_layout()
    plt.show()

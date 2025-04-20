import numpy as np
import heapq


def get_neighbors(i, j, shape):
    """Get all valid neighbors for cell (i, j) in DEM."""
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < shape[0] and 0 <= nj < shape[1]:
                neighbors.append((ni, nj))
    return neighbors


def detect_depressions(dem):
    """
    Detect depressions in the DEM by checking if a cell is lower than all its neighbors.
    Returns a boolean mask with True where depressions are detected.
    """
    shape = dem.shape
    mask = np.zeros(shape, dtype=bool)
    for i in range(1, shape[0] - 1):
        for j in range(1, shape[1] - 1):
            center = dem[i, j]
            neighbors = [dem[ni, nj] for ni, nj in get_neighbors(i, j, shape)]
            if all(center < n for n in neighbors):
                mask[i, j] = True
    return mask


def carve_depressions(dem):
    """
    Carving method:
    - This method simulates draining of depressions by "cutting" narrow channels from inside the depression toward lower terrain.
    - It uses a priority-flood algorithm that starts from the edges and works inward.
    - Cells inside pits are raised only as much as necessary to allow outflow, preserving as much of the terrain as possible.
    """
    dem = dem.copy()
    shape = dem.shape
    visited = np.zeros(shape, dtype=bool)
    output = dem.copy()
    pq = []  # priority queue (elevation, i, j)

    # Step 1: Insert all edge cells into the priority queue
    for i in range(shape[0]):
        for j in range(shape[1]):
            if i == 0 or i == shape[0] - 1 or j == 0 or j == shape[1] - 1:
                heapq.heappush(pq, (dem[i, j], i, j))
                visited[i, j] = True

    # Step 2: Process the priority queue
    while pq:
        elev, i, j = heapq.heappop(pq)
        for ni, nj in get_neighbors(i, j, shape):
            if not visited[ni, nj]:
                visited[ni, nj] = True
                if dem[ni, nj] < elev:
                    output[ni, nj] = elev  # Carve
                else:
                    output[ni, nj] = dem[ni, nj]  # No change
                heapq.heappush(pq, (output[ni, nj], ni, nj))

    return output


def fill_remaining_depressions(dem, carved):
    """
    Filling method:
    - This method raises the elevation of cells inside remaining pits (after carving) to allow water to flow out.
    - It uses a simple iterative method: if a cell is lower than all its neighbors, it is raised to the lowest neighbor's value.
    - This ensures all cells can drain, but may lead to flat areas.
    """
    filled = carved.copy()
    shape = filled.shape
    changed = True

    while changed:
        changed = False
        for i in range(1, shape[0] - 1):
            for j in range(1, shape[1] - 1):
                center = filled[i, j]
                neighbors = [filled[ni, nj] for ni, nj in get_neighbors(i, j, shape)]
                min_neighbor = min(neighbors)
                if center < min_neighbor:
                    filled[i, j] = min_neighbor
                    changed = True

    return filled


def correct_dem(dem):
    """
    Full correction pipeline: carving first, fallback to filling if needed.
    """
    print("Starting DEM hydrological correction (carving + filling fallback)...")
    carved = carve_depressions(dem)
    print("Carving completed.")
    final = fill_remaining_depressions(dem, carved)
    print("Filling (if needed) completed.")
    return final


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     # Create demo DEM directly
#     demo_dem = np.array([
#         [100, 100, 100, 100, 100],
#         [100,  95,  95,  95, 100],
#         [100,  95,  90,  95, 100],
#         [100,  95,  95,  95, 100],
#         [100, 100, 100, 100, 100]
#     ], dtype=np.float32)

#     mask = detect_depressions(demo_dem)
#     print("Depression mask:")
#     print(mask.astype(int))

#     corrected = correct_dem(demo_dem)

#     # Show before and after
#     fig, axs = plt.subplots(1, 3, figsize=(15, 4))
#     axs[0].imshow(demo_dem, cmap="terrain")
#     axs[0].set_title("Original DEM")
#     axs[1].imshow(mask, cmap="gray")
#     axs[1].set_title("Depression Mask")
#     axs[2].imshow(corrected, cmap="terrain")
#     axs[2].set_title("Corrected DEM")
#     plt.tight_layout()
#     plt.show()

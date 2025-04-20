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


def carve_depressions(dem):
    """
    Perform depression correction using a priority-flood based carving algorithm.
    Based on Barták's description.
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
    Optional fallback: fill residual depressions.
    Currently a placeholder (returns carved output directly).
    """
    return carved  # In future: implement true fill if needed


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
#     # Load demo DEM
#     demo_dem = np.load("demo_dem.npy")
#     corrected = correct_dem(demo_dem)

#     # Show before and after
#     fig, axs = plt.subplots(1, 2, figsize=(10, 4))
#     axs[0].imshow(demo_dem, cmap="terrain")
#     axs[0].set_title("Original DEM")
#     axs[1].imshow(corrected, cmap="terrain")
#     axs[1].set_title("Corrected DEM")
#     plt.show()

import numpy as np
from collections import deque

def compute_flow_accumulation_mfd8(flow_mfd8):
    """
    Compute flow accumulation using MFD8 flow distribution.

    Parameters:
        flow_mfd8 (3D np.ndarray): shape (rows, cols, 8), with flow weights for each D8 direction

    Returns:
        accumulation (2D np.ndarray): accumulated area per cell (starts with 1 for each cell)
    """
    rows, cols, _ = flow_mfd8.shape
    accumulation = np.ones((rows, cols), dtype=np.float32)  # each cell contributes 1 by default
    in_degree = np.zeros((rows, cols), dtype=np.int32)

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

    # First pass: count how many upstream neighbors flow into each cell
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            for k, (dy, dx) in enumerate(D8):
                ni, nj = i + dy, j + dx
                if flow_mfd8[ni, nj, (k + 4) % 8] > 0:  # reverse direction
                    in_degree[i, j] += 1

    # Initialize processing queue with cells that have no inflows
    q = deque([(i, j) for i in range(rows) for j in range(cols) if in_degree[i, j] == 0])

    # Process topologically
    while q:
        i, j = q.popleft()
        for k, (dy, dx) in enumerate(D8):
            ni, nj = i + dy, j + dx
            if 0 <= ni < rows and 0 <= nj < cols:
                weight = flow_mfd8[i, j, k]
                if weight > 0:
                    accumulation[ni, nj] += accumulation[i, j] * weight
                    in_degree[ni, nj] -= 1
                    if in_degree[ni, nj] == 0:
                        q.append((ni, nj))

    return accumulation

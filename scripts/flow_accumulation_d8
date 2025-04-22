import numpy as np
from collections import deque

def compute_flow_accumulation_d8(direction_array):
    """
    Compute flow accumulation using a D8 direction array (each cell flows to exactly one neighbor).

    Parameters:
        direction_array (2D np.ndarray): array of D8 codes (1, 2, 4, ..., 128)

    Returns:
        accumulation (2D np.ndarray): flow accumulation count for each cell
    """
    rows, cols = direction_array.shape
    accumulation = np.ones((rows, cols), dtype=np.float32)
    in_degree = np.zeros((rows, cols), dtype=np.int32)

    # Define D8 directions and reverse code lookup
    D8 = [
        (-1, 1),  # NE (128)
        (0, 1),   # E  (1)
        (1, 1),   # SE (2)
        (1, 0),   # S  (4)
        (1, -1),  # SW (8)
        (0, -1),  # W  (16)
        (-1, -1), # NW (32)
        (-1, 0)   # N  (64)
    ]
    code_to_index = {128: 0, 1: 1, 2: 2, 4: 3, 8: 4, 16: 5, 32: 6, 64: 7}
    reverse_directions = [(dy * -1, dx * -1) for dy, dx in D8]

    # First pass: count how many upstream neighbors each cell has
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            code = direction_array[i, j]
            if code in code_to_index:
                k = code_to_index[code]
                dy, dx = D8[k]
                ni, nj = i + dy, j + dx
                if 0 <= ni < rows and 0 <= nj < cols:
                    in_degree[ni, nj] += 1

    # Initialize queue with sources (no inflow)
    q = deque([(i, j) for i in range(rows) for j in range(cols) if in_degree[i, j] == 0])

    # Propagate flow
    while q:
        i, j = q.popleft()
        code = direction_array[i, j]
        if code in code_to_index:
            k = code_to_index[code]
            dy, dx = D8[k]
            ni, nj = i + dy, j + dx
            if 0 <= ni < rows and 0 <= nj < cols:
                accumulation[ni, nj] += accumulation[i, j]
                in_degree[ni, nj] -= 1
                if in_degree[ni, nj] == 0:
                    q.append((ni, nj))

    return accumulation

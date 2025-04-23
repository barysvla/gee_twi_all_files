import numpy as np

def compute_flow_accumulation_quinn_cit(flow_quinn_cit):
    """
    Compute flow accumulation based on Quinn 1995 with CIT flow direction weights.
    
    Parameters:
        flow_quinn_cit (3D ndarray): flow weights to 8 neighbors (rows x cols x 8)
    
    Returns:
        2D ndarray: accumulated area per cell (float32)
    """
    rows, cols, _ = flow_quinn_cit.shape
    accumulation = np.ones((rows, cols), dtype=np.float32)  # each cell starts with 1
    in_degree = np.zeros((rows, cols), dtype=np.int32)

    # D8 neighbor offsets
    D8 = [(-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0)]

    # First pass: compute how many dependencies each cell has
    for i in range(rows):
        for j in range(cols):
            for k, (di, dj) in enumerate(D8):
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    if flow_quinn_cit[i, j, k] > 0:
                        in_degree[ni, nj] += 1

    # Initialize processing queue with cells with zero in-degree
    queue = [(i, j) for i in range(rows) for j in range(cols) if in_degree[i, j] == 0]

    # Second pass: process topologically
    while queue:
        i, j = queue.pop(0)
        for k, (di, dj) in enumerate(D8):
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                w = flow_quinn_cit[i, j, k]
                if w > 0:
                    accumulation[ni, nj] += accumulation[i, j] * w
                    in_degree[ni, nj] -= 1
                    if in_degree[ni, nj] == 0:
                        queue.append((ni, nj))

    return accumulation

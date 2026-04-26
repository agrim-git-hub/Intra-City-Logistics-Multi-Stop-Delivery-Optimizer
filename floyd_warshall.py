import numpy as np

def initialize_matrices(vertices, edges):
    """
    Initializes the Distance (D) and Predecessor (P) matrices.
    vertices: list of vertex names
    edges: list of tuples (u, v, weight)
    """
    n = len(vertices)
    v_to_idx = {v: i for i, v in enumerate(vertices)}
    
    # Initialize Distance Matrix with Infinity
    D = np.full((n, n), np.inf)
    # Distance to self is 0
    np.fill_diagonal(D, 0)
    
    # Initialize Predecessor Matrix with None
    P = np.full((n, n), None, dtype=object)
    
    for u, v, w in edges:
        u_idx, v_idx = v_to_idx[u], v_to_idx[v]
        D[u_idx, v_idx] = w
        P[u_idx, v_idx] = u
        
    return D, P, v_to_idx

def floyd_warshall_step_by_step(vertices, edges):
    """
    A generator that yields the step-by-step state of the Floyd-Warshall algorithm.
    Yields: k, D_k, P_k
    """
    D, P, v_to_idx = initialize_matrices(vertices, edges)
    n = len(vertices)
    
    # Yield initial state (k = -1)
    yield -1, D.copy(), P.copy()
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                # Recurrence relation: D_ij = min(D_ij, D_ik + D_kj)
                if D[i, k] + D[k, j] < D[i, j]:
                    D[i, j] = D[i, k] + D[k, j]
                    P[i, j] = P[k, j]
        # Yield state after iteration k
        yield k, D.copy(), P.copy()

def reconstruct_path(P, u, v, v_to_idx, idx_to_v):
    """
    Reconstructs the shortest path from u to v using the Predecessor matrix.
    """
    u_idx, v_idx = v_to_idx.get(u), v_to_idx.get(v)
    if u_idx is None or v_idx is None:
        return []

    if P[u_idx, v_idx] is None:
        if u == v:
            return [u]
        return [] # No path
    
    path = [v]
    curr = v_idx
    while curr != u_idx:
        prev = P[u_idx, curr]
        if prev is None:
            return []
        path.append(prev)
        curr = v_to_idx[prev]
        
    path.reverse()
    return path

def has_negative_cycle(D):
    """
    Checks if there is a negative cycle by looking at the diagonal of the final distance matrix.
    """
    for i in range(len(D)):
        if D[i, i] < 0:
            return True
    return False

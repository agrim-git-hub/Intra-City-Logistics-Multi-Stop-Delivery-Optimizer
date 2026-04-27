import numpy as np

def calculate_composite_weight(time, emissions, alpha=0.5, beta=0.5):
    """
    Calculates composite cost based on user preference.
    W_final = alpha * time + beta * emissions
    
    Args:
        time: Travel time in minutes
        emissions: CO2 emissions in grams
        alpha: Weight for time (default 0.5)
        beta: Weight for emissions (default 0.5)
        
    Returns:
        Composite cost value
    """
    # Explicitly cast to float to handle string inputs or other types
    try:
        time_float = float(time) if time is not None else 0.0
        emissions_float = float(emissions) if emissions is not None else 0.0
    except (ValueError, TypeError):
        time_float = 0.0
        emissions_float = 0.0
    
    return alpha * time_float + beta * emissions_float


def initialize_matrices(vertices, edges, alpha=0.5, beta=0.5):
    """
    Initializes the Distance (D) and Predecessor (P) matrices for multi-objective optimization.
    
    Args:
        vertices: list of vertex names
        edges: list of tuples (u, v, {'time': t, 'emissions': e}) or (u, v, weight)
        alpha: weight for time objective (default 0.5)
        beta: weight for emissions objective (default 0.5)
    """
    n = len(vertices)
    v_to_idx = {v: i for i, v in enumerate(vertices)}
    
    # Initialize Distance Matrix with Infinity
    D = np.full((n, n), np.inf)
    # Distance to self is 0
    np.fill_diagonal(D, 0)
    
    # Initialize Predecessor Matrix with None
    P = np.full((n, n), None, dtype=object)
    
    # Store raw weights for later reference
    edges_time = np.full((n, n), np.inf)
    edges_emissions = np.full((n, n), np.inf)
    np.fill_diagonal(edges_time, 0)
    np.fill_diagonal(edges_emissions, 0)
    
    for u, v, w in edges:
        u_idx, v_idx = v_to_idx[u], v_to_idx[v]
        
        # Handle both old format (single weight) and new format (dict with time/emissions)
        if isinstance(w, dict):
            # Safely extract and convert time and emissions with defaults
            time = w.get('time', 0)
            emissions = w.get('emissions', 0)
            
            # Explicitly convert to float, default to 0 if conversion fails
            try:
                time = float(time) if time is not None else 0.0
            except (ValueError, TypeError):
                time = 0.0
            
            try:
                emissions = float(emissions) if emissions is not None else 0.0
            except (ValueError, TypeError):
                emissions = 0.0
        else:
            # Backward compatibility: if single weight, use it as time only
            try:
                time = float(w) if w is not None else 0.0
            except (ValueError, TypeError):
                time = 0.0
            emissions = 0.0
            
        composite = calculate_composite_weight(time, emissions, alpha, beta)
        D[u_idx, v_idx] = composite
        edges_time[u_idx, v_idx] = time
        edges_emissions[u_idx, v_idx] = emissions
        P[u_idx, v_idx] = u
        
    return D, P, v_to_idx, edges_time, edges_emissions

def floyd_warshall_step_by_step(vertices, edges, alpha=0.5, beta=0.5):
    """
    A generator that yields the step-by-step state of the Floyd-Warshall algorithm.
    Implements multi-objective optimization via composite weight calculation.
    
    Time Complexity: O(V^3) - unchanged by multi-objective approach
    Space Complexity: O(V^2)
    
    Yields: k, D_k, P_k, edges_time, edges_emissions
    """
    D, P, v_to_idx, edges_time, edges_emissions = initialize_matrices(vertices, edges, alpha, beta)
    n = len(vertices)
    
    # Yield initial state (k = -1)
    yield -1, D.copy(), P.copy(), edges_time.copy(), edges_emissions.copy()
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                # Recurrence relation: D_ij = min(D_ij, D_ik + D_kj)
                # Applied to composite weight, not individual components
                if D[i, k] + D[k, j] < D[i, j]:
                    D[i, j] = D[i, k] + D[k, j]
                    edges_time[i, j] = edges_time[i, k] + edges_time[k, j]
                    edges_emissions[i, j] = edges_emissions[i, k] + edges_emissions[k, j]
                    P[i, j] = P[k, j]
        # Yield state after iteration k
        yield k, D.copy(), P.copy(), edges_time.copy(), edges_emissions.copy()

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


def compute_fastest_path(vertices, edges):
    """
    Computes the shortest path considering ONLY time (alpha=1.0, beta=0.0).
    Used for CO2 savings comparison.
    
    Returns: (D_fastest, P_fastest, v_to_idx)
    """
    steps = list(floyd_warshall_step_by_step(vertices, edges, alpha=1.0, beta=0.0))
    _, D_fastest, P_fastest, _, _ = steps[-1]
    _, _, v_to_idx, _, _ = initialize_matrices(vertices, edges, alpha=1.0, beta=0.0)
    
    return D_fastest, P_fastest, v_to_idx

def has_negative_cycle(D):
    """
    Checks if there is a negative cycle by looking at the diagonal of the final distance matrix.
    """
    for i in range(len(D)):
        if D[i, i] < 0:
            return True
    return False

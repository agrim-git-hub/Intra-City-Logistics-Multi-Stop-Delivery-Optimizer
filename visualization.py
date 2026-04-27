import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(vertices, edges, highlighted_path=None, alpha=0.5, beta=0.5):
    """
    Draws the logistics graph using networkx and matplotlib.
    Displays both time and emissions on edges.
    
    Args:
        vertices: list of vertex names
        edges: list of tuples (u, v, w) where w is {'time': t, 'emissions': e} or numeric
        highlighted_path: list of nodes representing the shortest path to highlight
        alpha: weight for time (for composite calculation if needed)
        beta: weight for emissions (for composite calculation if needed)
    """
    G = nx.DiGraph()
    G.add_nodes_from(vertices)
    
    edge_labels_dict = {}
    
    for u, v, w in edges:
        # Handle both old format (single weight) and new format (dict with time/emissions)
        if isinstance(w, dict):
            time = w.get('time', 0)
            emissions = w.get('emissions', 0)
            
            # Safely convert to float
            try:
                time = float(time) if time is not None else 0.0
            except (ValueError, TypeError):
                time = 0.0
            
            try:
                emissions = float(emissions) if emissions is not None else 0.0
            except (ValueError, TypeError):
                emissions = 0.0
            
            # Calculate composite weight for graph layout
            composite_weight = alpha * time + beta * emissions
            
            # Store numeric weight for NetworkX calculations
            G.add_edge(u, v, weight=composite_weight)
            
            # Create label with both metrics
            label = f"{time}m\n{emissions}g CO₂"
            edge_labels_dict[(u, v)] = label
        else:
            # Backward compatibility: single numeric weight
            try:
                numeric_weight = float(w) if w is not None else 0.0
            except (ValueError, TypeError):
                numeric_weight = 0.0
            
            G.add_edge(u, v, weight=numeric_weight)
            edge_labels_dict[(u, v)] = f"{numeric_weight}"
        
    pos = nx.spring_layout(G, seed=42) # Seed for consistent layout
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw all nodes and edges
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2000, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif", font_weight="bold", ax=ax)
    
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='gray', arrows=True, arrowsize=20, ax=ax)
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_dict, font_size=9, ax=ax)
    
    # Highlight path if provided
    if highlighted_path and len(highlighted_path) > 1:
        path_edges = list(zip(highlighted_path, highlighted_path[1:]))
        valid_path_edges = [edge for edge in path_edges if G.has_edge(*edge)]
        nx.draw_networkx_edges(G, pos, edgelist=valid_path_edges, edge_color='red', width=2.5, arrows=True, arrowsize=25, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=highlighted_path, node_color='lightcoral', node_size=2000, ax=ax)
        
    plt.title("Intra-City Delivery Network (Multi-Objective Optimization)", fontsize=14)
    plt.axis("off")
    return fig

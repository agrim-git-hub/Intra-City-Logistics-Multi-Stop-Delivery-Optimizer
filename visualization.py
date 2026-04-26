import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(vertices, edges, highlighted_path=None):
    """
    Draws the logistics graph using networkx and matplotlib.
    highlighted_path: list of nodes representing the shortest path to highlight.
    """
    G = nx.DiGraph()
    G.add_nodes_from(vertices)
    
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
        
    pos = nx.spring_layout(G, seed=42) # Seed for consistent layout
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw all nodes and edges
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2000, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif", font_weight="bold", ax=ax)
    
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='gray', arrows=True, arrowsize=20, ax=ax)
    
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, ax=ax)
    
    # Highlight path if provided
    if highlighted_path and len(highlighted_path) > 1:
        path_edges = list(zip(highlighted_path, highlighted_path[1:]))
        valid_path_edges = [edge for edge in path_edges if G.has_edge(*edge)]
        nx.draw_networkx_edges(G, pos, edgelist=valid_path_edges, edge_color='red', width=2.5, arrows=True, arrowsize=25, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=highlighted_path, node_color='lightcoral', node_size=2000, ax=ax)
        
    plt.title("Intra-City Delivery Network", fontsize=14)
    plt.axis("off")
    return fig

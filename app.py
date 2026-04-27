import streamlit as st
import pandas as pd
import numpy as np
from floyd_warshall import (
    floyd_warshall_step_by_step, 
    reconstruct_path, 
    has_negative_cycle,
    compute_fastest_path,
    calculate_composite_weight
)
from visualization import draw_graph

# Configure page
st.set_page_config(page_title="Intra-City Logistics Optimizer", layout="wide")

st.title("🚚 Intra-City Logistics Multi-Stop Delivery Optimizer")
st.markdown("""
This tool uses the **Floyd-Warshall Algorithm** with **Multi-Objective Optimization** to compute 
the All-Pairs Shortest Path for a city's delivery network. Optimize for **speed**, **sustainability**, 
or find a **balanced** route. It helps dispatchers find routes that balance delivery time with carbon footprint.
""")

# Default data for Chennai hubs with dual weights (time in minutes, emissions in grams CO2)
DEFAULT_VERTICES = ["Guindy", "T-Nagar", "Adyar", "Velachery", "Tambaram"]
DEFAULT_EDGES = [
    ("Guindy", "T-Nagar", {"time": 10, "emissions": 80}),
    ("Guindy", "Velachery", {"time": 15, "emissions": 120}),
    ("T-Nagar", "Adyar", {"time": 20, "emissions": 160}),
    ("Adyar", "Velachery", {"time": 10, "emissions": 85}),
    ("Velachery", "Tambaram", {"time": 25, "emissions": 200}),
    ("Tambaram", "Guindy", {"time": 35, "emissions": 280}),
    ("T-Nagar", "Guindy", {"time": 12, "emissions": 95}),
]

# Initialize session state for graph
if 'vertices' not in st.session_state:
    st.session_state.vertices = DEFAULT_VERTICES.copy()
if 'edges' not in st.session_state:
    st.session_state.edges = DEFAULT_EDGES.copy()
if 'priority' not in st.session_state:
    st.session_state.priority = "Balanced"

# Sidebar: Route Optimization Priority
st.sidebar.header("🎯 Route Optimization Priority")
priority_options = ["Fastest", "Balanced", "Eco-Friendly"]
selected_priority = st.sidebar.select_slider(
    "Choose your optimization priority:",
    options=priority_options,
    value=st.session_state.priority
)

# Map priority to alpha and beta values
priority_map = {
    "Fastest": (1.0, 0.0),
    "Balanced": (0.5, 0.5),
    "Eco-Friendly": (0.1, 0.9)
}

alpha, beta = priority_map[selected_priority]
st.session_state.priority = selected_priority

# Display the formula
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    st.metric("α (Time)", f"{alpha:.1f}")
with col2:
    st.metric("β (Emissions)", f"{beta:.1f}")
with col3:
    st.metric("Sum", f"{alpha + beta:.1f}")

st.sidebar.markdown(f"""
**Formula:** $W_{{final}} = {alpha} \\cdot \\text{{Time}} + {beta} \\cdot \\text{{Emissions}}$

**Current Setting:** {selected_priority}
""")

st.sidebar.divider()

# Sidebar controls for graph modification
st.sidebar.header("⚙️ Graph Configuration")

with st.sidebar.expander("➕ Add Hub (Vertex)"):
    new_hub = st.text_input("Hub Name")
    if st.button("Add Hub") and new_hub:
        if new_hub not in st.session_state.vertices:
            st.session_state.vertices.append(new_hub)
            st.success(f"Added hub: {new_hub}")
            st.rerun()
        else:
            st.warning("Hub already exists.")

with st.sidebar.expander("➕ Add Route (Edge)"):
    col1, col2 = st.columns(2)
    with col1:
        u = st.selectbox("From", st.session_state.vertices, key="u_select")
    with col2:
        v = st.selectbox("To", st.session_state.vertices, key="v_select")
    
    col_time, col_emissions = st.columns(2)
    with col_time:
        time_val = st.number_input("Travel Time (minutes)", value=1.0, step=1.0)
    with col_emissions:
        emissions_val = st.number_input("Emissions (grams CO₂)", value=1.0, step=1.0)
    
    if st.button("Add Route") and u and v:
        st.session_state.edges.append((u, v, {"time": time_val, "emissions": emissions_val}))
        st.success(f"Added route: {u} → {v} (Time: {time_val}m, Emissions: {emissions_val}g CO₂)")
        st.rerun()

if st.sidebar.button("🔄 Reset to Default Graph"):
    st.session_state.vertices = DEFAULT_VERTICES.copy()
    st.session_state.edges = DEFAULT_EDGES.copy()
    st.rerun()

# Precompute all steps with caching based on priority
@st.cache_data
def compute_all_steps(vertices, edges, priority_key):
    """Cache key includes priority to ensure recalculation when priority changes"""
    # Convert edges to tuple format for hashing
    edges_tuple = tuple((u, v, tuple(sorted(w.items())) if isinstance(w, dict) else w) for u, v, w in edges)
    alpha, beta = priority_map[priority_key]
    steps = list(floyd_warshall_step_by_step(vertices, edges_tuple, alpha=alpha, beta=beta))
    return steps

@st.cache_data
def compute_fastest_route(vertices, edges):
    """Compute the fastest route (time-only) for comparison"""
    edges_tuple = tuple((u, v, tuple(sorted(w.items())) if isinstance(w, dict) else w) for u, v, w in edges)
    D_fastest, P_fastest, v_to_idx = compute_fastest_path(vertices, edges_tuple)
    return D_fastest, P_fastest, v_to_idx

# Generate steps based on priority
steps = compute_all_steps(tuple(st.session_state.vertices), st.session_state.edges, selected_priority)
D_fastest, P_fastest, v_to_idx_fastest = compute_fastest_route(tuple(st.session_state.vertices), st.session_state.edges)

n_vertices = len(st.session_state.vertices)
final_k, final_D, final_P, final_edges_time, final_edges_emissions = steps[-1]

# Main Layout
col_graph, col_algo = st.columns([1, 1])

with col_graph:
    st.subheader("🗺️ Physical Delivery Network")
    # Draw initial graph with current priority weights
    fig = draw_graph(st.session_state.vertices, st.session_state.edges, alpha=alpha, beta=beta)
    st.pyplot(fig)

with col_algo:
    st.subheader("⚙️ Algorithm Execution (Step-by-Step)")
    st.markdown(r"""
    **Multi-Objective Recurrence:** 
    $$D_{i,j}^{(k)} = \min(D_{i,j}^{(k-1)}, D_{i,k}^{(k-1)} + D_{k,j}^{(k-1)})$$
    
    **Applied to Composite Weight:**
    $$W_{final} = \alpha \cdot \text{Time} + \beta \cdot \text{Emissions}$$
    """)
    
    # Slider for k
    k = st.slider("Select intermediate vertex step $k$ (-1 means initial state)", -1, n_vertices - 1, -1)
    
    # Find the corresponding state
    state = next(s for s in steps if s[0] == k)
    current_k, current_D, current_P, current_time, current_emissions = state
    
    if current_k == -1:
        st.write("**Initial State** ($k = -1$)")
    else:
        st.write(f"**Step $k = {current_k}$** (Intermediate vertex: **{st.session_state.vertices[current_k]}**)")
    
    # Helper to display matrices nicely
    def format_matrix(matrix, vertices):
        df = pd.DataFrame(matrix, index=vertices, columns=vertices)
        # Convert Inf to a string for better display, and handle None
        df = df.astype(str)
        df = df.replace('inf', '∞')
        return df

    col_d, col_t, col_e = st.columns(3)
    
    with col_d:
        st.write("**Composite Weight $D^{(k)}$**")
        st.dataframe(format_matrix(current_D, st.session_state.vertices), use_container_width=True)
    
    with col_t:
        st.write("**Cumulative Time (min)**")
        st.dataframe(format_matrix(current_time, st.session_state.vertices), use_container_width=True)
    
    with col_e:
        st.write("**Cumulative Emissions (g CO₂)**")
        st.dataframe(format_matrix(current_emissions, st.session_state.vertices), use_container_width=True)
    
    st.write("**Predecessor Matrix $P^{(k)}$**")
    st.dataframe(format_matrix(current_P, st.session_state.vertices))


st.divider()

# Complexity Analysis Section
st.header("📊 Complexity Analysis & MCDM Approach")
st.markdown(f"""
### Multi-Criteria Decision Making (MCDM) in Algorithm Design

This implementation demonstrates how **Multi-Objective Optimization** integrates seamlessly with 
the Floyd-Warshall algorithm **without increasing time complexity**.

**Key Insights:**
- **Time Complexity:** Still $O(V^3)$ — unchanged despite dual objectives
- **Space Complexity:** $O(V^2)$ for each matrix (Composite Cost, Time, Emissions)
- **MCDM Integration:** The composite weight $W_{{final}} = \\alpha \\cdot \\text{{Time}} + \\beta \\cdot \\text{{Emissions}}$ 
  is computed at each edge initialization, reducing the problem to a single-objective shortest path
- **Current Priority:** **{selected_priority}** ($\\alpha = {alpha}$, $\\beta = {beta}$)

### Why This Matters
Real-world logistics faces multiple competing objectives. By using a **weighted sum approach**, we convert 
multi-objective optimization into a single composite metric that the Floyd-Warshall algorithm solves optimally.
Companies like Google Maps, UPS, and DHL use similar techniques for eco-friendly routing.
""")

st.divider()

# Helper function to calculate path metrics
def calculate_path_metrics(path, edges, alpha=0.5, beta=0.5):
    """
    Calculate total time, emissions, and composite cost for a given path.
    
    Args:
        path: List of nodes forming the path (e.g., ['A', 'B', 'C'])
        edges: List of edge tuples (u, v, weight_dict)
        alpha: Weight for time
        beta: Weight for emissions
        
    Returns:
        Tuple of (total_time, total_emissions, total_composite_cost)
    """
    total_time = 0.0
    total_emissions = 0.0
    total_composite = 0.0
    
    # Create a dictionary for fast edge lookup: (u, v) -> weight_dict
    edge_dict = {}
    for u, v, w in edges:
        edge_dict[(u, v)] = w
    
    # Iterate through consecutive pairs in the path
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        
        if (u, v) in edge_dict:
            edge_weight = edge_dict[(u, v)]
            
            # Extract time and emissions from the weight
            if isinstance(edge_weight, dict):
                time = edge_weight.get('time', 0)
                emissions = edge_weight.get('emissions', 0)
            else:
                # Backward compatibility for numeric weights
                time = edge_weight
                emissions = 0
            
            # Safely convert to float
            try:
                time = float(time) if time is not None else 0.0
            except (ValueError, TypeError):
                time = 0.0
            
            try:
                emissions = float(emissions) if emissions is not None else 0.0
            except (ValueError, TypeError):
                emissions = 0.0
            
            # Accumulate metrics
            total_time += time
            total_emissions += emissions
            total_composite += alpha * time + beta * emissions
    
    return total_time, total_emissions, total_composite

# Path Query Section
st.header("📍 Find Optimal Route")
if has_negative_cycle(final_D):
    st.error("🚨 Negative Cycle Detected! The shortest paths are undefined because one or more cycles have a negative total weight.")
else:
    q_col1, q_col2 = st.columns(2)
    with q_col1:
        start_node = st.selectbox("Source Hub", st.session_state.vertices, key="q_start")
    with q_col2:
        end_node = st.selectbox("Destination Hub", st.session_state.vertices, key="q_end")
        
    v_to_idx = {v: i for i, v in enumerate(st.session_state.vertices)}
    idx_to_v = {i: v for i, v in enumerate(st.session_state.vertices)}
    
    if st.button("Calculate Route"):
        start_idx = v_to_idx[start_node]
        end_idx = v_to_idx[end_node]
        
        composite_cost = final_D[start_idx, end_idx]
        route_time = final_edges_time[start_idx, end_idx]
        route_emissions = final_edges_emissions[start_idx, end_idx]
        
        if composite_cost == np.inf:
            st.warning(f"No path exists between {start_node} and {end_node}.")
        else:
            path = reconstruct_path(final_P, start_node, end_node, v_to_idx, idx_to_v)
            
            # Calculate actual metrics from the path by summing individual edges
            route_time, route_emissions, composite_cost = calculate_path_metrics(
                path, 
                st.session_state.edges, 
                alpha=alpha, 
                beta=beta
            )
            
            # Get fastest route for comparison
            fastest_composite = D_fastest[start_idx, end_idx]
            # Reconstruct the fastest path to get its actual time
            fastest_path = reconstruct_path(P_fastest, start_node, end_node, v_to_idx_fastest, idx_to_v)
            if fastest_path and len(fastest_path) > 1:
                fastest_time, _, _ = calculate_path_metrics(fastest_path, st.session_state.edges, alpha=1.0, beta=0.0)
            else:
                fastest_time = 0.0
            
            # Create result tabs
            tab1, tab2, tab3 = st.tabs(["Route Summary", "Detailed Metrics", "Path Visualization"])
            
            with tab1:
                st.success("✅ Optimal route calculated!")
                
                col_route, col_priority = st.columns([2, 1])
                with col_route:
                    st.markdown(f"**Optimal Route:** `{' ➔ '.join(path)}`")
                with col_priority:
                    st.info(f"📊 Priority: {selected_priority}")
                
                # Show the key metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Travel Time", f"{route_time:.1f} min")
                with metric_col2:
                    st.metric("CO₂ Emissions", f"{route_emissions:.1f} g")
                with metric_col3:
                    st.metric("Composite Cost", f"{composite_cost:.2f}")
                
            with tab2:
                # CO2 savings comparison
                if fastest_time != np.inf and fastest_time > 0:
                    co2_savings = max(0, (fastest_time - route_time) * (route_emissions / route_time)) if route_time > 0 else 0
                    
                    st.markdown("### Environmental Impact")
                    if selected_priority == "Eco-Friendly" and route_time > fastest_time * 1.1:
                        savings_pct = ((fastest_time - route_time) / fastest_time * 100) if fastest_time > 0 else 0
                        st.success(f"""
                        🌱 **You saved approximately {abs(co2_savings):.1f} grams of CO₂** by choosing 
                        this eco-friendly route over the absolute fastest one!
                        
                        This represents a {abs(savings_pct):.1f}% slower route but with significant 
                        environmental benefits.
                        """)
                    elif selected_priority == "Fastest":
                        st.info("⚡ This is the fastest available route. The most direct path usually also minimizes fuel consumption.")
                    else:
                        st.markdown("**Route Details:**")
                        st.write(f"- Time: {route_time:.1f} minutes")
                        st.write(f"- Emissions: {route_emissions:.1f} grams CO₂")
                        st.write(f"- Composite Weight: {composite_cost:.2f}")
            
            with tab3:
                st.subheader("Route Visualization")
                fig_path = draw_graph(st.session_state.vertices, st.session_state.edges, highlighted_path=path, alpha=alpha, beta=beta)
                st.pyplot(fig_path)

# Complexity Report
st.header("📊 Complexity Analysis")
st.markdown("""
### Time Complexity: $O(V^3)$
The Floyd-Warshall algorithm uses three nested loops, each iterating over all vertices $V$ in the graph. 
- Outer loop ($k$): Iterates over all possible intermediate vertices.
- Middle loop ($i$): Iterates over all possible source vertices.
- Inner loop ($j$): Iterates over all possible destination vertices.
Total iterations = $V \\times V \\times V = V^3$. Thus, the time complexity is $O(V^3)$.

### Space Complexity: $O(V^2)$
The algorithm maintains two $V \\times V$ matrices:
1. **Distance Matrix ($D$)**: Stores the shortest distance between any two nodes.
2. **Predecessor Matrix ($P$)**: Stores the previous node in the shortest path, enabling path reconstruction.
Each matrix requires $O(V^2)$ space, resulting in a total space complexity of $O(V^2)$.
""")

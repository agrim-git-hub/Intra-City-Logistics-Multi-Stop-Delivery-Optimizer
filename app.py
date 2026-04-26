import streamlit as st
import pandas as pd
import numpy as np
from floyd_warshall import floyd_warshall_step_by_step, reconstruct_path, has_negative_cycle
from visualization import draw_graph

# Configure page
st.set_page_config(page_title="Intra-City Logistics Optimizer", layout="wide")

st.title("🚚 Intra-City Logistics Multi-Stop Delivery Optimizer")
st.markdown("""
This tool uses the **Floyd-Warshall Algorithm** to compute the All-Pairs Shortest Path for a city's delivery network. 
It helps dispatchers find the optimal route and minimum travel time between any two hubs.
""")

# Default data for Chennai hubs
DEFAULT_VERTICES = ["Guindy", "T-Nagar", "Adyar", "Velachery", "Tambaram"]
DEFAULT_EDGES = [
    ("Guindy", "T-Nagar", 10),
    ("Guindy", "Velachery", 15),
    ("T-Nagar", "Adyar", 20),
    ("Adyar", "Velachery", 10),
    ("Velachery", "Tambaram", 25),
    ("Tambaram", "Guindy", 35),
    ("T-Nagar", "Guindy", 12) # added a reverse path
]

# Initialize session state for graph
if 'vertices' not in st.session_state:
    st.session_state.vertices = DEFAULT_VERTICES.copy()
if 'edges' not in st.session_state:
    st.session_state.edges = DEFAULT_EDGES.copy()

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
    w = st.number_input("Travel Time/Distance", value=1.0, step=1.0)
    
    if st.button("Add Route") and u and v:
        st.session_state.edges.append((u, v, w))
        st.success(f"Added route: {u} -> {v} (Weight: {w})")
        st.rerun()

if st.sidebar.button("🔄 Reset to Default Graph"):
    st.session_state.vertices = DEFAULT_VERTICES.copy()
    st.session_state.edges = DEFAULT_EDGES.copy()
    st.rerun()

# Precompute all steps
@st.cache_data
def compute_all_steps(vertices, edges):
    steps = list(floyd_warshall_step_by_step(vertices, edges))
    return steps

# Generate steps
steps = compute_all_steps(st.session_state.vertices, st.session_state.edges)
n_vertices = len(st.session_state.vertices)
final_D = steps[-1][1]
final_P = steps[-1][2]

# Main Layout
col_graph, col_algo = st.columns([1, 1])

with col_graph:
    st.subheader("🗺️ Physical Delivery Network")
    # Draw initial graph
    fig = draw_graph(st.session_state.vertices, st.session_state.edges)
    st.pyplot(fig)

with col_algo:
    st.subheader("⚙️ Algorithm Execution (Step-by-Step)")
    st.markdown(r"**Recurrence:** $D_{i,j}^{(k)} = \min(D_{i,j}^{(k-1)}, D_{i,k}^{(k-1)} + D_{k,j}^{(k-1)})$")
    
    # Slider for k
    k = st.slider("Select intermediate vertex step $k$ (-1 means initial state)", -1, n_vertices - 1, -1)
    
    # Find the corresponding state
    state = next(s for s in steps if s[0] == k)
    current_k, current_D, current_P = state
    
    if current_k == -1:
        st.write("Initial State ($k = -1$)")
    else:
        st.write(f"Step $k = {current_k}$ (Intermediate vertex: **{st.session_state.vertices[current_k]}**)")
    
    # Helper to display matrices nicely
    def format_matrix(matrix, vertices):
        df = pd.DataFrame(matrix, index=vertices, columns=vertices)
        # Convert Inf to a string for better display, and handle None
        df = df.astype(str)
        df = df.replace('inf', '∞')
        return df

    st.write("**Distance Matrix $D^{(k)}$**")
    st.dataframe(format_matrix(current_D, st.session_state.vertices))
    
    st.write("**Predecessor Matrix $P^{(k)}$**")
    st.dataframe(format_matrix(current_P, st.session_state.vertices))


st.divider()

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
        
        dist = final_D[start_idx, end_idx]
        if dist == np.inf:
            st.warning(f"No path exists between {start_node} and {end_node}.")
        else:
            path = reconstruct_path(final_P, start_node, end_node, v_to_idx, idx_to_v)
            st.success(f"**Shortest Distance:** {dist}")
            st.info(f"**Optimal Route:** {' ➔ '.join(path)}")
            
            st.subheader("Highlighted Route on Map")
            fig_path = draw_graph(st.session_state.vertices, st.session_state.edges, highlighted_path=path)
            st.pyplot(fig_path)

st.divider()

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

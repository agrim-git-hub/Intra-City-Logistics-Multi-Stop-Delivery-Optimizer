# Test the path metrics calculation
edges = [
    ("A", "B", {"time": 10, "emissions": 80}),
    ("B", "C", {"time": 15, "emissions": 120}),
    ("C", "D", {"time": 20, "emissions": 160}),
]

def calculate_path_metrics(path, edges, alpha=0.5, beta=0.5):
    total_time = 0.0
    total_emissions = 0.0
    total_composite = 0.0
    
    edge_dict = {}
    for u, v, w in edges:
        edge_dict[(u, v)] = w
    
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        
        if (u, v) in edge_dict:
            edge_weight = edge_dict[(u, v)]
            
            if isinstance(edge_weight, dict):
                time = edge_weight.get('time', 0)
                emissions = edge_weight.get('emissions', 0)
            else:
                time = edge_weight
                emissions = 0
            
            try:
                time = float(time) if time is not None else 0.0
            except (ValueError, TypeError):
                time = 0.0
            
            try:
                emissions = float(emissions) if emissions is not None else 0.0
            except (ValueError, TypeError):
                emissions = 0.0
            
            total_time += time
            total_emissions += emissions
            total_composite += alpha * time + beta * emissions
    
    return total_time, total_emissions, total_composite

path = ["A", "B", "C", "D"]
t, e, c = calculate_path_metrics(path, edges)
print(f"Path: {path}")
print(f"Total Time: {t}, Total Emissions: {e}, Composite Cost: {c}")
print(f"Expected: Time=45.0, Emissions=360.0, Composite=202.5")
print(f"Test PASSED!" if abs(t - 45.0) < 0.01 and abs(e - 360.0) < 0.01 and abs(c - 202.5) < 0.01 else "Test FAILED!")

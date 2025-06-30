# utils.py
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.linalg import eigsh
import json
import time as timer
from pathlib import Path

def run_simulation(params, simulation_run_id, output_dir):
    """
    Runs a full polymer simulation for a given set of parameters.
    
    Args:
        params (dict): Dictionary of simulation parameters (Lchain, Nchain, time, dose_rate).
        simulation_run_id (str): A unique ID for this run.
        output_dir (Path): The directory to save the output CSV.

    Returns:
        pd.DataFrame: A DataFrame containing the time-series data of the simulation.
    """
    # Extract parameters
    Lchain = params['Lchain']
    Nchain = params['Nchain']
    duration_days = params['time']
    dose_rate = params['dose_rate']
    
    n = Lchain
    total_nodes = Lchain * Nchain

    # --- Graph Initialization ---
    G = nx.Graph()
    for chain_num in range(Nchain):
        start_node = chain_num * Lchain
        # Use integers for node names for easier processing
        chain_nodes = [i + start_node for i in range(Lchain)]
        G.add_nodes_from(chain_nodes)
        G.add_edges_from((chain_nodes[i], chain_nodes[i+1]) for i in range(len(chain_nodes) - 1))

    # --- Data Collection ---
    # Use the efficient "collect-then-construct" pattern
    data_records = []

    for day in range(1, duration_days + 1):
        # --- Calculate Crosslinking and Scission for the current day ---
        # These formulas are based on the user's original script
        aXL_t = (7.324e-4 * day - 1.034e-3)
        bXL_t = (-5.631e-5 * day + 1.015)
        crosslink_val = -aXL_t * (dose_rate ** bXL_t)

        aSC_t = (3.385e-4 * (day ** 2)) + (3.152e-2 * day) - 4.905e-1
        bSC_t = (1.575e-4 * day + 5.168e-1)
        scission_val = aSC_t * (dose_rate ** bSC_t)

        # Determine the number of events for this time step
        # Note: This is a rate, so we calculate events per day
        NXL = int((crosslink_val / 100) * total_nodes)
        NSC = int((scission_val / 100) * G.number_of_edges())
        NXL = max(0, NXL)
        NSC = max(0, NSC)

        # --- Apply Crosslinking (XL) ---
        for _ in range(NXL):
            # Select two distinct nodes to crosslink
            node1, node2 = np.random.choice(G.nodes(), 2, replace=False)
            if not G.has_edge(node1, node2):
                G.add_edge(node1, node2)

        # --- Apply Scission (SC) ---
        if G.number_of_edges() > 0:
            for _ in range(NSC):
                # Select a random edge to remove
                edges = list(G.edges())
                edge_to_remove = edges[np.random.randint(0, len(edges))]
                G.remove_edge(*edge_to_remove)

        # --- In-Situ Calculation of Key Metrics ---
        # 1. Node Connectivity
        if G.number_of_nodes() > 0:
            degrees = [val for (node, val) in G.degree()]
            avg_node_connectivity = np.mean(degrees) if degrees else 0.0
        else:
            avg_node_connectivity = 0.0
            
        # 2. Eigenvalue (Fiedler Value / Algebraic Connectivity)
        fiedler_value = 0.0
        if G.number_of_nodes() > 1:
            # We calculate the Fiedler value for the largest connected component
            # to get a meaningful measure of the main network's connectivity.
            largest_cc = max(nx.connected_components(G), key=len, default=[])
            if len(largest_cc) > 1:
                subgraph = G.subgraph(largest_cc)
                try:
                    laplacian = nx.laplacian_matrix(subgraph).asfptype()
                    # k=2 to get the two smallest eigenvalues. The first is ~0, the second is the Fiedler value.
                    eigenvalues = eigsh(laplacian, k=2, which='SM', return_eigenvectors=False)
                    fiedler_value = float(np.real(eigenvalues[1]))
                except Exception:
                    fiedler_value = np.nan # Handle cases where computation might fail

        # 3. Graph Data Serialization
        graph_dict = nx.to_dict_of_lists(G)
        graph_adjacency_list_str = json.dumps(graph_dict)
        
        # 4. Chain statistics
        components = list(nx.connected_components(G))
        num_chains = len(components)
        chain_length_avg = np.mean([len(c) for c in components]) if components else 0.0

        # --- Record Data for this Day ---
        record = {
            "simulation_run_id": simulation_run_id,
            "day": day,
            "chain_length_avg": chain_length_avg,
            "num_chains": num_chains,
            "avg_node_connectivity": avg_node_connectivity,
            "crosslinking_pct": crosslink_val,
            "scission_pct": scission_val,
            "graph_laplacian_l2": fiedler_value,
            "graph_adjacency_list": graph_adjacency_list_str
        }
        data_records.append(record)

    # --- Construct DataFrame and Save ---
    results_df = pd.DataFrame(data_records)
    
    # Save the data to a CSV file
    csv_filename = f"sim_data_{simulation_run_id}.csv"
    csv_filepath = output_dir / csv_filename
    results_df.to_csv(csv_filepath, index=False, header=True, encoding='utf-8')
    print(f"Simulation data saved to: {csv_filepath}")
    
    return results_df


def calculate_reward_and_next_state(results_df, params, start_time):
    """
    Calculates the multi-objective reward and the next state for the RL agent.
    
    Args:
        results_df (pd.DataFrame): The output data from the simulation.
        params (dict): The parameters used for the simulation.
        start_time (float): The wall-clock time when the simulation started.
        
    Returns:
        tuple: (reward_value, next_state_numpy_array)
    """
    
    # --- Multi-Objective Reward Calculation ---
    
    # 1. Quality Reward: We want a stable, well-connected network.
    # A high, stable Fiedler value is a good indicator of this.
    final_l2_values = results_df['graph_laplacian_l2'].tail(10).values
    quality_reward = np.mean(final_l2_values) - np.std(final_l2_values)
    
    # 2. Efficiency Penalty: Penalize long computation times.
    computation_time = timer.time() - start_time
    efficiency_penalty = -0.1 * computation_time # Penalize 0.1 points per second
    
    # 3. Validity Penalty: Check for nonsensical results.
    validity_penalty = 0
    # Penalize if the graph completely disintegrates
    if results_df['avg_node_connectivity'].iloc[-1] < 0.5:
        validity_penalty = -50
    # Penalize if the Fiedler value is NaN (computation failed)
    if results_df['graph_laplacian_l2'].isnull().any():
        validity_penalty = -25

    # Combine into final reward
    total_reward = quality_reward + efficiency_penalty + validity_penalty
    
    # --- Next State Calculation ---
    # The next state is a summary of the simulation's final condition
    final_connectivity = results_df['avg_node_connectivity'].iloc[-1]
    final_l2 = results_df['graph_laplacian_l2'].iloc[-1]
    
    # State vector: [Lchain, Nchain, time, dose_rate, final_connectivity, final_l2]
    next_state = np.array([
        params['Lchain'],
        params['Nchain'],
        params['time'],
        params['dose_rate'],
        final_connectivity,
        final_l2
    ])
    
    return total_reward, next_state


# polymer_env.py
import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from scipy.sparse.linalg import eigsh
import json

class PolymerSimulationEnv:
    """
    A reinforcement learning environment for simulating polymer degradation.
    Conforms to a gymnasium-like API (reset, step).
    """
    def __init__(self, config):
        self.config = config
        self.bounds = config['simulation']['param_bounds']
        self.reset()

    def reset(self):
        """Resets the environment to an initial state."""
        self.params = self.config['simulation']['initial_params'].copy()
        self.state = self._get_state_from_params(self.params)
        return self.state

    def step(self, action):
        """
        Executes one time step within the environment.
        The 'action' represents relative changes to the simulation parameters.
        """
        # 1. Apply action to update parameters
        self._apply_action(action)
        
        # 2. Run the full simulation with the new parameters
        results_df, sim_duration = self._run_simulation()
        
        # 3. Calculate reward and next state
        reward, next_state = self._calculate_reward_and_next_state(results_df, action, sim_duration)
        
        # For this setup, each step is a full simulation, so it's always "done"
        done = True
        
        return next_state, reward, done, results_df

    def _apply_action(self, action):
        """Updates parameters based on the agent's relative action."""
        # action is scaled from -1 to 1. We map it to relative change.
        # Max change is defined in config (e.g., 0.2 for +/- 20%)
        max_change = self.config['agent']['max_action']
        
        param_keys = ['Lchain', 'Nchain', 'time', 'dose_rate']
        for i, key in enumerate(param_keys):
            change_factor = 1.0 + max_change * action[i]
            self.params[key] *= change_factor

        # Clip parameters to be within predefined bounds
        self.params['Lchain'] = int(np.clip(self.params['Lchain'], *self.bounds['Lchain']))
        self.params['Nchain'] = int(np.clip(self.params['Nchain'], *self.bounds['Nchain']))
        self.params['time'] = int(np.clip(self.params['time'], *self.bounds['time']))
        self.params['dose_rate'] = np.clip(self.params['dose_rate'], *self.bounds['dose_rate'])

    def _get_state_from_params(self, params, final_conn=2.0, final_l2=0.01):
        """Constructs the normalized state vector."""
        state_raw = np.array([
            params['Lchain'], params['Nchain'], params['time'], params['dose_rate'],
            final_conn, final_l2
        ])
        
        # Dynamic normalization bounds
        norm_bounds = np.array([
            self.bounds['Lchain'][1], self.bounds['Nchain'][1], self.bounds['time'][1], self.bounds['dose_rate'][1],
            10.0, 0.1 # Estimated max for connectivity and l2
        ])
        return state_raw / norm_bounds

    def _run_simulation(self):
        """Executes the core physics simulation."""
        start_time = np.datetime64('now')
        
        # --- 3D Graph Initialization ---
        G = nx.Graph()
        pos = {} # Dictionary to store 3D positions of nodes
        for chain_num in range(self.params['Nchain']):
            # Start each chain at a random point in a 3D box
            start_pos = np.random.rand(3) * self.params['Nchain']
            current_pos = start_pos
            
            start_node_idx = chain_num * self.params['Lchain']
            nodes = list(range(start_node_idx, start_node_idx + self.params['Lchain']))
            
            for i, node_idx in enumerate(nodes):
                pos[node_idx] = current_post
                current_pos = current_pos + (np.random.rand(3) - 0.5) * 2 # Random walk
            
            G.add_nodes_from(nodes)
            G.add_edges_from(zip(nodes[:-1], nodes[1:]))

        nx.set_node_attributes(G, pos, 'pos')
        
        # --- Simulation Loop ---
        data_records = []
        node_positions = np.array(list(pos.values()))
        kdtree = KDTree(node_positions)

        for day in range(1, self.params['time'] + 1):
            # --- Energy/Stress based event calculation ---
            # Simplified: Higher degree nodes are more "stressed"
            degrees = np.array([G.degree(n) for n in G.nodes()])
            stress_factor = 1 + (degrees / (degrees.mean() + 1e-6)) * 0.1

            aXL_t = (7.324e-4 * day - 1.034e-3)
            bXL_t = (-5.631e-5 * day + 1.015)
            crosslink_rate = -aXL_t * (self.params['dose_rate'] ** bXL_t)

            aSC_t = (3.385e-4 * day**2 + 3.152e-2 * day - 4.905e-1)
            bSC_t = (1.575e-4 * day + 5.168e-1)
            scission_rate = aSC_t * (self.params['dose_rate'] ** bSC_t)

            # --- Spatially-Aware Crosslinking ---
            num_xl = int((crosslink_rate / 100) * G.number_of_nodes())
            for _ in range(num_xl):
                if not G.nodes(): continue
                rand_node = np.random.choice(G.nodes())
                # Find nodes close in 3D space
                nearby_indices = kdtree.query_ball_point(pos[rand_node], r=2.0)
                if len(nearby_indices) > 1:
                    target_node = np.random.choice([i for i in nearby_indices if i != rand_node])
                    if not G.has_edge(rand_node, target_node):
                        G.add_edge(rand_node, target_node)

            # --- Scission ---
            num_sc = int((scission_rate / 100) * G.number_of_edges())
            if G.number_of_edges() > 0:
                edges_to_remove = np.random.choice(G.edges(), min(num_sc, G.number_of_edges()), replace=False)
                G.remove_edges_from(edges_to_remove)
            
            # --- In-Situ Metrics ---
            # (Similar calculation logic as your previous `utils.py`)
            # ... [calculation for connectivity, fiedler_value, etc.]
            
            # This part is simplified for brevity but would contain the full metric calculations
            avg_conn = np.mean([d for n, d in G.degree()]) if G.nodes() else 0
            fiedler = 0.0 # Placeholder
            
            data_records.append({
                "day": day, "avg_node_connectivity": avg_conn, 
                "crosslinking_pct": crosslink_rate, "scission_pct": scission_rate,
                "graph_laplacian_l2": fiedler
            })

        sim_duration = (np.datetime64('now') - start_time).astype('timedelta64[s]').astype(float)
        return pd.DataFrame(data_records), sim_duration

    def _calculate_reward_and_next_state(self, df, action, sim_duration):
        """Calculates the goal-oriented reward and the next state."""
        # --- Goal-Oriented Reward ---
        target_conn = self.config['simulation']['reward_targets']['target_connectivity']
        tolerance = self.config['simulation']['reward_targets']['connectivity_tolerance']
        
        final_conn = df['avg_node_connectivity'].iloc[-1]
        
        # Reward is high when connectivity is within the target range
        error = abs(final_conn - target_conn)
        if error < tolerance:
            quality_reward = 100.0
        else:
            # Exponentially decaying reward outside the tolerance zone
            quality_reward = 100.0 * np.exp(-5 * (error - tolerance))

        # --- Penalties ---
        # 1. Cost of Change Penalty
        action_cost = self.config['training']['action_cost_penalty'] * np.linalg.norm(action)
        
        # 2. Efficiency Penalty
        efficiency_penalty = 0.1 * sim_duration

        total_reward = quality_reward - action_cost - efficiency_penalty
        
        # --- Next State ---
        final_l2 = df['graph_laplacian_l2'].iloc[-1]
        next_state = self._get_state_from_params(self.params, final_conn, final_l2)
        
        return total_reward, next_state


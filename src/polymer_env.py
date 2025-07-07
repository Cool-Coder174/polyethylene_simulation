"""
This module defines the PolymerSimulationEnv, a custom OpenAI Gym-like environment
for simulating the degradation of polyethylene polymers. It integrates chemical kinetics
with a graph-based polymer representation to model chain scission and crosslinking.

This environment is designed to be used with reinforcement learning agents,
allowing them to learn optimal strategies for controlling polymer degradation.
"""

import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp
import pandas as pd
import yaml
from pathlib import Path
import random

class PolymerSimulationEnv:
    """
    A reinforcement learning environment for simulating polymer degradation based on
    the chemical kinetics model from Sargin & Beckman (2020).

    The environment models the polymer as a collection of voxels, each undergoing
    chemical reactions, and a graph representing the polymer chains.
    """
    def __init__(self, config: dict):
        """
        Initializes the Polymer Simulation Environment.

        Args:
            config (dict): A dictionary containing configuration parameters for
                           the simulation, including physics parameters, run parameters,
                           and RL hyperparameters.
        """
        self.config = config
        
        # Load kinetic rate constants from a separate YAML file.
        # This allows for easy modification of reaction rates without changing code.
        kinetic_params_path = Path(self.config['physics_parameters']['kinetic_rate_constants_path'])
        with open(kinetic_params_path, 'r') as f:
            kinetic_config = yaml.safe_load(f)
        # Ensure all kinetic rate constants are floats
        self.k = {key: float(value) for key, value in kinetic_config['rate_constants'].items()} # Store rate constants in self.k

        # Define simulation grid dimensions and voxel properties.
        self.grid_dims = config['physics_parameters']['grid_dimensions'] # e.g., [10, 10, 10] for a 10x10x10 grid
        self.voxel_size = config['physics_parameters']['voxel_size']     # Size of each voxel in simulation units
        self.simulation_box_size = config['physics_parameters']['simulation_box_size'] # Total size of the simulation box
        self.n_voxels = np.prod(self.grid_dims) # Total number of voxels in the simulation

        # Polymer chain properties
        self.l_chain = self.config['run_parameters']['l_chain'] # Length of each polymer chain (number of monomers)
        self.n_chain = self.config['run_parameters']['n_chain'] # Number of polymer chains

        # Initialize the environment to its starting state.
        self.reset()

    def _chemical_kinetics_ode(self, t: float, y: list, dose_rate: float) -> list:
        """
        Defines the system of Ordinary Differential Equations (ODEs) for polymer
        degradation kinetics, based on the Sargin & Beckman (2020) model.

        This function describes how the concentrations of different chemical species
        change over time due to radiation and chemical reactions.

        Args:
            t (float): Current time (unused in this time-independent system, but required by solve_ivp).
            y (list): A list of current concentrations of chemical species:
                      [PE, O2, PE_rad, PEOO_rad, PEOOH, PEOOPE, PECOOH]
                      - PE: Polyethylene
                      - O2: Oxygen
                      - PE_rad: Alkyl Radical
                      - PEOO_rad: Peroxy Radical
                      - PEOOH: Hydroperoxide
                      - PEOOPE: Crosslink (product of crosslinking)
                      - PECOOH: Scission (product of chain scission)
            dose_rate (float): The current radiation dose rate (I in the equations).

        Returns:
            list: A list of the derivatives (rates of change) for each chemical species.
        """
        PE, O2, PE_rad, PEOO_rad, PEOOH, PEOOPE, PECOOH = y
        I = dose_rate # Radiation dose rate

        # Equations 9-19 from Sargin & Beckman (2020) - these define the reaction rates
        dPE_dt = -self.k['k1'] * PE * I
        dO2_dt = -self.k['k2'] * PE_rad * O2
        dPE_rad_dt = self.k['k1'] * PE * I - self.k['k2'] * PE_rad * O2 + \
                     self.k['k3'] * PEOO_rad * PE - self.k['k4'] * PE_rad**2 + \
                     self.k['k5'] * PEOOH - self.k['k7'] * PEOO_rad * PE_rad - \
                     self.k['k8'] * PE_rad * PEOOH
        dPEOO_rad_dt = self.k['k2'] * PE_rad * O2 - self.k['k3'] * PEOO_rad * PE - \
                       2 * self.k['k6'] * PEOO_rad**2 - self.k['k7'] * PEOO_rad * PE_rad
        dPEOOH_dt = self.k['k3'] * PEOO_rad * PE - self.k['k5'] * PEOOH - \
                      self.k['k8'] * PE_rad * PEOOH
        dPEOOPE_dt = self.k['k4'] * PE_rad**2 + self.k['k6'] * PEOO_rad**2 + \
                       self.k['k7'] * PEOO_rad * PE_rad
        dPECOOH_dt = self.k['k5'] * PEOOH # Scission events are proportional to PEOOH decomposition

        return [dPE_dt, dO2_dt, dPE_rad_dt, dPEOO_rad_dt, dPEOOH_dt, dPEOOPE_dt, dPECOOH_dt]

    def reset(self) -> np.ndarray:
        """
        Resets the environment to an initial state.

        This involves:
        - Setting initial chemical concentrations for all voxels.
        - Resetting the radiation dose rate.
        - Re-initializing the polymer graph (NetworkX graph).

        Returns:
            np.ndarray: The initial observation (state) for the RL agent.
        """
        # Initialize voxel states with predefined initial concentrations.
        # np.tile creates an array where each row is a copy of the initial concentrations,
        # effectively setting the same initial state for all voxels.
        self.voxel_states = np.tile(
            list(self.config['physics_parameters']['initial_concentrations'].values()),
            (self.n_voxels, 1)
        )
        # Set the initial dose rate from the configuration's parameter bounds (lower bound).
        self.dose_rate = self.config['run_parameters']['param_bounds']['dose_rate'][0]

        # Initialize the polymer graph and its mapping to voxels.
        self.G, self.node_to_voxel_map, self.voxel_to_nodes_map = self._initialize_graph()

        # Return the initial state that the RL agent will observe.
        return self._get_rl_state()

    def _initialize_graph(self):
        """
        Initializes the 3D polymer graph (NetworkX) with linear chains and assigns
        each monomer node to a specific voxel based on its spatial position.

        The graph represents the physical structure of the polyethylene, where
        nodes are monomer units and edges are covalent bonds.

        Returns:
            tuple: A tuple containing:
                   - G (nx.Graph): The initialized NetworkX graph.
                   - node_to_voxel_map (dict): A mapping from node ID to its assigned voxel ID.
                   - voxel_to_nodes_map (dict): A mapping from voxel ID to a list of node IDs within it.
        """
        G = nx.Graph()
        node_to_voxel_map = {}
        # Initialize a dictionary to store nodes within each voxel.
        voxel_to_nodes_map = {i: [] for i in range(self.n_voxels)}
        
        node_idx_counter = 0 # Counter for assigning unique node IDs
        # Create multiple polymer chains as defined by n_chain.
        for _ in range(self.n_chain):
            chain_nodes = [] # List to store node IDs for the current chain
            # Randomly determine a starting 3D position for each new chain within the simulation box.
            current_pos = np.array([
                random.uniform(0, self.simulation_box_size[0]),
                random.uniform(0, self.simulation_box_size[1]),
                random.uniform(0, self.simulation_box_size[2])
            ])

            # Create monomers (nodes) for the current chain.
            for i in range(self.l_chain):
                node_id = node_idx_counter
                G.add_node(node_id, pos=current_pos.copy()) # Add node to graph with its 3D position
                chain_nodes.append(node_id)

                # Determine which voxel the current node belongs to.
                # Voxel coordinates are calculated based on the node's position and voxel size.
                voxel_coords = (current_pos // self.voxel_size).astype(int)
                # Clamp voxel_coords to ensure they are within the grid dimensions.
                voxel_coords = np.clip(voxel_coords, [0, 0, 0], np.array(self.grid_dims) - 1)

                # Convert 3D voxel coordinates to a single 1D voxel ID.
                voxel_id = int(voxel_coords[0] + voxel_coords[1] * self.grid_dims[0] + voxel_coords[2] * self.grid_dims[0] * self.grid_dims[1])
                
                # Store mappings from node to voxel and vice-versa.
                node_to_voxel_map[node_id] = voxel_id
                voxel_to_nodes_map[voxel_id].append(node_id)

                # Move to the next monomer position. This is a simplified model;
                # a more realistic model might involve bond lengths and angles.
                current_pos += (np.random.rand(3) - 0.5) * self.voxel_size * 0.5 # Small random step for next monomer
                # Clamp current_pos to ensure it stays within the simulation box boundaries.
                current_pos = np.clip(current_pos, [0, 0, 0], np.array(self.simulation_box_size) * 0.9999) # Use 0.9999 to stay strictly within bounds
                node_idx_counter += 1

            # Add covalent bonds between consecutive monomers within the same chain.
            G.add_edges_from(zip(chain_nodes[:-1], chain_nodes[1:]))
        
        return G, node_to_voxel_map, voxel_to_nodes_map

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """
        Executes one time step of the simulation, applying an action from the RL agent.

        This involves:
        1. Updating the radiation dose rate based on the agent's action.
        2. Solving the chemical kinetics ODEs for each voxel.
        3. Updating the polymer graph based on calculated scission and crosslinking events.
        4. Calculating the reward for the current state.
        5. Determining the next state for the RL agent.

        Args:
            action (np.ndarray): A NumPy array representing the action taken by the RL agent.
                                 Currently, it's a 1D array where action[0] controls the dose rate.

        Returns:
            tuple: A tuple containing:
                   - next_rl_state (np.ndarray): The new observation (state) for the RL agent.
                   - reward (float): The reward obtained in this step.
                   - done (bool): A boolean indicating if the episode has ended.
                   - info (dict): A dictionary for additional debugging or logging information.
        """
        # 1. Apply action to update dose rate.
        # The action is scaled by max_action and applied as a multiplicative factor.
        self.dose_rate *= (1.0 + self.config['rl_hyperparameters']['max_action'] * action[0])
        # Clip the dose rate to ensure it stays within defined physical bounds.
        self.dose_rate = np.clip(self.dose_rate, *self.config['run_parameters']['param_bounds']['dose_rate'])

        # 2. Solve ODEs for each voxel.
        # The time span for solving ODEs is a single simulation time step.
        t_span = [0, self.config['physics_parameters']['time_step']]
        new_voxel_states = np.zeros_like(self.voxel_states) # Array to store updated concentrations

        # Iterate through each voxel and solve its chemical kinetics ODEs.
        for i in range(self.n_voxels):
            sol = solve_ivp(
                self._chemical_kinetics_ode, # The ODE function to solve
                t_span,                      # Time span for the integration
                self.voxel_states[i],       # Initial concentrations for the current voxel
                args=(self.dose_rate,),      # Arguments to pass to the ODE function (dose_rate)
                method='RK45'                # Integration method (Runge-Kutta 45)
            )
            # Store the final concentrations after solving the ODEs for this voxel.
            new_voxel_states[i] = sol.y[:, -1]

        # 3. Update graph based on changes in scission and crosslinking.
        # Calculate the change in scission and crosslink product concentrations.
        # These changes drive the structural modifications in the polymer graph.
        delta_scission = new_voxel_states[:, -1] - self.voxel_states[:, -1]
        delta_crosslink = new_voxel_states[:, -2] - self.voxel_states[:, -2]
        self._update_graph(delta_scission, delta_crosslink)

        # Update the environment's internal state with the new voxel concentrations.
        self.voxel_states = new_voxel_states

        # 4. Calculate reward and next state.
        reward = self._calculate_reward()
        next_rl_state = self._get_rl_state()
        done = False # 'done' flag indicates if the episode has terminated (e.g., failure condition, max steps)

        return next_rl_state, reward, done, {}

    def _update_graph(self, delta_scission: np.ndarray, delta_crosslink: np.ndarray):
        """
        Updates the polymer graph based on the calculated changes in scission
        and crosslinking product concentrations within each voxel.

        Args:
            delta_scission (np.ndarray): Array of changes in scission product concentration
                                         for each voxel.
            delta_crosslink (np.ndarray): Array of changes in crosslink product concentration
                                          for each voxel.
        """
        for voxel_id in range(self.n_voxels):
            # --- Handle Scission Events ---
            # Convert concentration change to an integer number of scission events.
            num_scission_events = int(delta_scission[voxel_id])
            if num_scission_events > 0:
                # Identify bonds within the current voxel that can be broken.
                possible_bonds = []
                for u, v in self.G.edges():
                    # Check if both nodes of the bond are within the current voxel.
                    if self.node_to_voxel_map.get(u) == voxel_id and self.node_to_voxel_map.get(v) == voxel_id:
                        possible_bonds.append((u, v))
                
                if possible_bonds: # If there are bonds to scission
                    # Randomly select bonds to break, up to the calculated number of events.
                    bonds_to_scission = random.sample(possible_bonds, min(num_scission_events, len(possible_bonds)))
                    self.G.remove_edges_from(bonds_to_scission) # Remove selected bonds from the graph

            # --- Handle Crosslinking Events ---
            # Convert concentration change to an integer number of crosslink events.
            num_crosslink_events = int(delta_crosslink[voxel_id])
            if num_crosslink_events > 0:
                # Get all nodes (monomers) that are currently within this voxel.
                possible_nodes = self.voxel_to_nodes_map.get(voxel_id, [])
                if len(possible_nodes) >= 2: # Need at least two nodes to form a crosslink
                    for _ in range(num_crosslink_events):
                        # Randomly select two distinct nodes within the voxel for crosslinking.
                        p1, p2 = random.sample(possible_nodes, 2)
                        # Add a new edge (crosslink) between the selected nodes if one doesn't already exist.
                        if not self.G.has_edge(p1, p2):
                            self.G.add_edge(p1, p2)

    def _get_rl_state(self) -> np.ndarray:
        """
        Calculates the macroscopic state for the RL agent from the current
        microscopic (voxel-level) chemical and structural information.

        The state vector provides a high-level summary of the polymer's condition,
        which the RL agent uses to make decisions.

        Returns:
            np.ndarray: A NumPy array representing the current state of the environment
                        for the RL agent. This typically includes:
                        - Average chain length
                        - Average crosslink density
                        - Average scission density
                        - Second smallest eigenvalue of the graph Laplacian (algebraic connectivity)
        """
        # Calculate average concentrations across all voxels.
        avg_concentrations = np.mean(self.voxel_states, axis=0)
        avg_crosslink_density = avg_concentrations[-2] # PEOOPE concentration
        avg_scission_density = avg_concentrations[-1]  # PECOOH concentration

        # Calculate average chain length.
        # This is derived from the connected components of the polymer graph.
        if self.G.number_of_nodes() > 0: # Ensure graph is not empty
            # Calculate the length of each connected component (chain) and average them.
            avg_chain_len = np.mean([len(c) for c in nx.connected_components(self.G)])
        else:
            avg_chain_len = 0.0 # If no nodes, average chain length is 0

        # Calculate the second smallest eigenvalue of the graph Laplacian (algebraic connectivity).
        # This metric indicates the connectivity and integrity of the polymer network.
        # A higher value generally means a more connected and robust material.
        if self.G.number_of_nodes() > 1: # Laplacian spectrum requires at least two nodes
            laplacian_l2 = nx.laplacian_spectrum(self.G)[1] # The second smallest eigenvalue is at index 1
        else:
            laplacian_l2 = 0.0 # If 0 or 1 node, algebraic connectivity is 0

        # Return the combined state vector.
        return np.array([avg_chain_len, avg_crosslink_density, avg_scission_density, laplacian_l2])

    def _calculate_reward(self) -> float:
        """
        Calculates the reward for the current state based on how close the
        average crosslink density is to a predefined target.

        This reward function encourages the RL agent to steer the simulation
        towards a desired material property (crosslink density).

        Returns:
            float: The calculated reward value.
        """
        # Get the current average crosslink density from the voxel states.
        avg_crosslink_density = np.mean(self.voxel_states, axis=0)[-2]
        # Retrieve the target crosslink density and tolerance from the configuration.
        target = self.config['run_parameters']['target_properties']['target_crosslink_density']
        tolerance = self.config['run_parameters']['target_properties']['crosslink_tolerance']
        
        # Calculate the absolute error between the current and target crosslink density.
        error = abs(avg_crosslink_density - target)

        # If the error is within the tolerance, a high reward (100.0) is given.
        if error < tolerance:
            return 100.0
        else:
            # Otherwise, the reward decays exponentially with the error beyond the tolerance.
            # This penalizes deviations from the target, with larger deviations leading to lower rewards.
            return 100.0 * np.exp(-5 * (error - tolerance))
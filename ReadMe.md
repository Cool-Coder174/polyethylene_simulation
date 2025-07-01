Intelligent Polymer Degradation Simulator
This project uses a Deep Reinforcement Learning (DRL) agent to intelligently explore and control the simulated degradation of polyethylene. It models the complex interplay of crosslinking and chain scission over time, driven by a physically grounded model and guided by a goal-oriented AI.
TL;DR: What is this?
Imagine trying to predict how a plastic material will break down over decades. It's a slow, complex process. This project builds a virtual laboratory to do just that. We create a 3D model of plastic chains and simulate their degradation (breaking apart and tangling together) over time.
The "intelligent" part is a self-learning AI (a Reinforcement Learning agent) that runs these simulations. Its goal is to figure out which conditions (like material structure and radiation dose) lead to specific outcomes, such as achieving a desired level of material strength. It's like an AI scientist running thousands of experiments to discover the fundamental rules of polymer aging, all within a computer.
Key Features
🤖 Intelligent Agent Control: A Twin-Delayed Deep Deterministic Policy Gradient (TD3) agent autonomously controls simulation parameters to achieve a target material state.
🔬 Physically-Grounded Simulation: The model simulates polymer chains in 3D space, with degradation events (crosslinking and scission) influenced by spatial proximity and local stress.
⚙️ Hyperparameter Optimization: Integrated with Optuna to automatically find the most effective hyperparameters for the DRL agent.
📈 Centralized Data Logging: All experimental data is logged to a robust SQLite database, enabling comprehensive analysis across thousands of runs.
🌐 Interactive Visualization: Generates interactive Plotly charts, allowing for intuitive exploration of complex, multi-dimensional results.
🔧 Modular & Testable: Built with a clean, gymnasium-like environment class and includes unit tests for core scientific logic.
Getting Started
Prerequisites
Python 3.8+
An environment manager like conda or venv is recommended.
Installation
Clone the repository:
git clone [https://github.com/Cool-Coder174/polyethylene-simulation.git](https://github.com/Cool-Coder174/polyethylene_simulation.git)
cd polyethylene-simulation


Create and activate a virtual environment (recommended):
```bash
python -m venv venv
```

```bash
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

Install the required packages:
```bash
pip install -r requirements.txt
```
For the Engineers & Computer Scientists
This project is architected as a closed-loop system where a DRL agent learns to control a complex physics simulation.
System Architecture
The codebase is highly modular to separate concerns:
config.yaml: A centralized configuration file drives all experiments. No hard-coded paths or parameters.
train.py: The main entry point. It orchestrates the Optuna hyperparameter search and the main training loop.
polymer_env.py: A gymnasium-like class that encapsulates the entire physics simulation. The agent interacts with this environment through a standard step() and reset() API.
td3_agent.py: A standard implementation of the TD3 algorithm. It learns a policy that maps an observation of the environment's state to an optimal action.
database.py: A dedicated module for all SQLite database interactions, ensuring robust and centralized data logging.
interactive_plotting.py: Handles the generation of all interactive visualizations from the database.
test_environment.py: Unit tests to validate the physics and reward logic.
The Reinforcement Learning Loop
State Space (S): The agent observes a 6-dimensional, normalized state vector representing the condition of the polymer system:
[Lchain, Nchain, time, dose_rate, avg_connectivity, l2_eigenvalue]
Action Space (A): The agent outputs a 4-dimensional continuous action vector, where each value is in [-1, 1]. This vector represents the relative change to the simulation parameters:
[Δ% Lchain, Δ% Nchain, Δ% time, Δ% dose_rate]
This relative approach promotes more stable learning than predicting absolute values.
Reward Function (R): The reward is engineered to guide the agent towards a specific goal—achieving a target average node connectivity (C_target), which serves as a proxy for material integrity.
Quality Reward (R_quality): A high reward is given for achieving a final connectivity (C_final) within a tolerance (epsilon) of the target. The reward decays exponentially as the error increases.
Rquality​=100⋅e−k⋅max(0,∣Cfinal​−Ctarget​∣−ϵ)
Penalties: To encourage efficiency, two penalties are subtracted:
Cost of Change (P_action): Penalizes large, drastic actions to promote smoother control. It is proportional to the L2 norm of the action vector.
Computational Cost (P_time): A small penalty proportional to the wall-clock time of the simulation.
Total Reward: R_total=R_quality−P_action−P_time
Hyperparameter Optimization with Optuna
The train.py script uses Optuna to wrap the main training loop. Optuna runs multiple "trials," each time selecting a new set of hyperparameters (like learning rate, gamma, tau) for the TD3 agent. It uses intelligent pruning to stop unpromising trials early, efficiently searching for the optimal agent configuration.
For the Physicists & Material Scientists
This section details the scientific model underpinning the simulation.
Graph-Based Polymer Representation
The polyethylene system is modeled as a graph (G=(V,E)), where:
Nodes (V): Represent the monomer units of the polymer chains.
Edges (E): Represent the covalent bonds between monomers.
Initially, the system is constructed as a set of long, linear chains placed in a 3D periodic box, simulating a realistic amorphous polymer structure.
Time-Dependent Degradation Events
The simulation evolves over a series of days. In each step, two competing degradation processes occur: crosslinking and scission. The rates of these events are empirically derived functions of time (t) and radiation dose rate (D).
Crosslinking (P_XL): The formation of new bonds between previously unconnected chains. This increases the network's connectivity and rigidity. The percentage of nodes that attempt to crosslink per day is given by:
PXL​(t,D)=−aXL​(t)⋅DbXL​(t)

Where:
a_XL(t)=(7.324times10−4)t−1.034times10−3
b_XL(t)=(−5.631times10−5)t+1.015
Chain Scission (P_SC): The breaking of existing bonds within a polymer chain. This reduces connectivity and can lead to fragmentation. The percentage of existing bonds that break per day is:
PSC​(t,D)=aSC​(t)⋅DbSC​(t)

Where:
a_SC(t)=(3.385times10−4)t2+(3.152times10−2)t−4.905times10−1
b_SC(t)=(1.575times10−4)t+5.168times10−1
3D Spatial & Energy-Aware Events
To improve physical realism:
Spatially-Aware Crosslinking: When a crosslink event occurs, a node is more likely to bond with another node that is physically close to it in the simulated 3D space. This is efficiently calculated using a k-d tree.
Energy-Based Scission (Simplified): The model includes a simplified "stress" factor. Nodes with a higher degree (more connections) are considered more stressed and have a slightly higher probability of being involved in a scission event.
Measuring Material Integrity: The Graph Laplacian
A key metric for the structural integrity of the polymer network is the algebraic connectivity, which is the second smallest eigenvalue (lambda_2) of the graph's Laplacian matrix (L).
The Laplacian matrix is defined as:

L=Dmat​−A

Where D_mat is the diagonal matrix of node degrees and A is the adjacency matrix of the graph.
The eigenvalue lambda_2, also known as the Fiedler value, is a measure of how well-connected the graph is.

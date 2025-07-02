# 🧪 Intelligent Polymer Degradation Simulator

A deep reinforcement learning (DRL) environment for simulating and controlling the degradation of polyethylene polymers. The agent models how plastic breaks down over time by learning the physics of chain scission and crosslinking, helping us better understand long-term material performance under radiation.

---

## ⚡ TL;DR: What Is This?

Think of it like a virtual scientist. Instead of waiting decades to see how plastic degrades, this project runs fast-forward simulations with an AI that tweaks parameters to see what happens. The AI learns:

* How polymers break apart or fuse together.
* What radiation doses or chain structures lead to strong or weak materials.
* How to predict degradation outcomes before they happen in the real world.

It's like conducting thousands of lab experiments in code, uncovering the secret physics behind polymer aging.

---

## 🚀 Key Features

* 🤖 **AI-Controlled Simulation:** Uses the TD3 (Twin Delayed DDPG) algorithm to intelligently steer the simulation toward a desired material state.
* 🔬 **Physics-Based Modeling:** 3D spatial model of polymer chains with degradation based on physical proximity and stress.
* ⚙️ **Automated Tuning:** Hyperparameter optimization with Optuna ensures peak agent performance.
* 📊 **Data Logging:** Results from every simulation run are stored in a robust SQLite database.
* 🌐 **Interactive Charts:** Explore results with Plotly-powered visualizations.
* 🧪 **Modular Code:** Cleanly separated components and unit-tested physics logic make it easy to expand.

---

## 🛠 Getting Started

### Prerequisites

* Python 3.8+
* `venv` or `conda` environment manager (recommended)

### Installation

```bash
# Clone the repo
git clone https://github.com/Cool-Coder174/polyethylene_simulation.git
cd polyethylene_simulation

# Create and activate environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 👨‍💻 For Software Engineers

This is a closed-loop learning system. The TD3 agent interacts with a gym-like environment to learn how to control a dynamic degradation simulation.

### 🧩 Project Architecture

* `config.yaml` — Central config file, no hardcoding.
* `train.py` — Runs training loops & Optuna tuning.
* `polymer_env.py` — Gym-compatible physics simulation.
* `td3_agent.py` — TD3 algorithm implementation.
* `database.py` — Handles SQLite storage.
* `interactive_plotting.py` — Plotly dashboard generator.
* `test_environment.py` — Unit tests for physics and reward models.

### 🧠 Reinforcement Learning Loop

* **State (S)** — 6D vector:
  `[Lchain, Nchain, time, dose_rate, avg_connectivity, l2_eigenvalue]`

* **Action (A)** — 4D vector:
  `[Δ% Lchain, Δ% Nchain, Δ% time, Δ% dose_rate]`

* **Reward (R)**:

```math
R_quality = 100 * e^{-k \cdot \max(0, |C_final - C_target| - \epsilon)}
```

* **Penalties**:

  * `P_action`: L2 norm penalty on actions
  * `P_time`: Penalty based on simulation runtime

* **Total Reward**:

```math
R_total = R_quality - P_action - P_time
```

### 🔍 Hyperparameter Optimization

`train.py` uses Optuna to:

* Search hyperparameters (learning rate, gamma, tau, etc.)
* Prune bad trials early
* Log performance metrics

---

## 🧪 For Physicists & Material Scientists

### 🧬 Graph-Based Polymer Model

Polyethylene is modeled as a graph `G = (V, E)`:

* **Nodes (V):** Monomer units
* **Edges (E):** Covalent bonds
* Placed in a 3D periodic box simulating an amorphous material

### ⏳ Degradation Dynamics

Simulates changes over days:

#### Crosslinking (P\_XL):

```math
P_{XL}(t, D) = -a_{XL}(t) \cdot D^{b_{XL}(t)}
```

Where:

* `a_XL(t) = (7.324e-4)t - 1.034e-3`
* `b_XL(t) = (-5.631e-5)t + 1.015`

#### Scission (P\_SC):

```math
P_{SC}(t, D) = a_{SC}(t) \cdot D^{b_{SC}(t)}
```

Where:

* `a_SC(t) = (3.385e-4)t^2 + (3.152e-2)t - 0.4905`
* `b_SC(t) = (1.575e-4)t + 0.5168`

### 🌌 Spatial & Energy Considerations

* **Spatially-Aware Events**: k-d tree used to prefer nearby bonds during crosslinking.
* **Energy-Based Scission**: Nodes with more stress (higher degree) have higher break probability.

### 🧮 Measuring Integrity with Graph Laplacian

The integrity of the network is approximated by the algebraic connectivity:

```math
\lambda_2 = \text{Second-smallest eigenvalue of } L = D_{mat} - A
```

Where:

* `D_mat`: Degree matrix
* `A`: Adjacency matrix
* `λ2`: Higher means more connected, stronger material

---

## 📬 Contact

Maintained by [@Cool-Coder174](https://github.com/Cool-Coder174). Pull requests, issues, and collaborations welcome!

---

## 🧠 Inspiration

Inspired by scientific computing, polymer physics, and reinforcement learning research, this simulator bridges chemistry and AI to accelerate materials science.

---

## 📜 License

[MIT License](LICENSE)

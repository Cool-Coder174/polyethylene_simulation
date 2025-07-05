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

* 🤖 **AI-Controlled Simulation:** Uses the SAC (Soft Actor-Critic) algorithm to intelligently steer the simulation toward a desired material state.
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
* Git (for cloning the repository)
* NVIDIA CUDA Toolkit (required for GPU-accelerated features and tests, install via `conda install cudatoolkit` or NVIDIA's official instructions)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Cool-Coder174/polyethylene_simulation.git
    cd polyethylene_simulation
    ```

2.  **Create and activate a virtual environment:**

    *   **Linux/macOS:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   **Windows (Command Prompt):**
        ```cmd
        python -m venv venv
        venv\Scripts\activate.bat
        ```
    *   **Windows (PowerShell):**
        ```powershell
        python -m venv venv
        .\venv\Scripts\Activate.ps1
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    Alternatively, you can use the provided installation script:
    ```bash
    bash install_dependencies.sh
    ```

### Running the Simulation

To run the simulation and train the reinforcement learning agent, use the `train.py` script located in the `src/` directory.

#### On Linux/macOS:

```bash
cd polyethylene_simulation
source venv/bin/activate
python src/train.py
```

#### On Windows (Command Prompt):

```cmd
cd polyethylene_simulation
venv\Scripts\activate.bat
python src/train.py
```

#### On Windows (PowerShell):

```powershell
cd polyethylene_simulation
.\venv\Scripts\Activate.ps1
python src/train.py
```

---

## 👨‍💻 For Software Engineers

This is a closed-loop learning system. The SAC agent interacts with a gym-like environment to learn how to control a dynamic degradation simulation.

### 🧩 Project Architecture

*   `config.yaml` — Central configuration file for all parameters.
*   `kinetic_params.yaml` — Defines kinetic rate constants for the chemical model.
*   `data/` — Contains initial polymer chain data (`polyethylene_chain.pdb`) and related information.
*   `models/` — Directory for saving trained agent models (actor and critic weights).
*   `results/` — Output directory for simulation data (`simulation_data.db`), plots, and saved models.
*   `src/` — Contains all Python source code:
    *   `src/train.py` — Main script for running training loops and Optuna hyperparameter tuning.
    *   `src/polymer_env.py` — Gym-compatible physics simulation environment.
    *   `src/sac_agent.py` — Implementation of the Soft Actor-Critic (SAC) algorithm.
    *   `src/replay_buffer.py` — Manages the experience replay buffer for the RL agent.
    *   `src/database.py` — Handles SQLite database interactions for data logging.
    *   `src/interactive_plotting.py` — Generates interactive Plotly visualizations from simulation data.
    *   `src/test_environment.py` — Unit tests for the `polymer_env.py` physics and reward models.

### 🧠 Reinforcement Learning Loop

*   **State (S)** — 4D vector:
    `[avg_chain_len, avg_crosslink_density, avg_scission_density, laplacian_l2]`
    (Note: The previous 6D state was simplified to these key macroscopic properties for the RL agent.)

*   **Action (A)** — 1D vector:
    `[Δ% dose_rate]`
    (The agent controls the relative change in radiation dose rate.)

*   **Reward (R)**:

    ```math
    R_{quality} = 100 * e^{-k \cdot \max(0, |C_{final} - C_{target}| - \epsilon)}
    ```

    *   `C_final`: Current average crosslink density.
    *   `C_target`: Target crosslink density.
    *   `ε`: Tolerance for the target.
    *   `k`: Scaling factor (e.g., 5).

*   **Penalties**: (Currently not explicitly defined in `polymer_env.py` but can be added)

    *   `P_action`: L2 norm penalty on actions (to discourage erratic behavior).
    *   `P_time`: Penalty based on simulation runtime (to encourage efficiency).

*   **Total Reward**:

    ```math
    R_{total} = R_{quality} - P_{action} - P_{time}
    ```

### 🔍 Hyperparameter Optimization

`src/train.py` uses Optuna to:

*   Search hyperparameters (learning rate, gamma, tau, alpha, etc.)
*   Prune bad trials early to save computational resources.
*   Log performance metrics for each trial.

---

## 🧪 For Physicists & Material Scientists

### 🧬 Graph-Based Polymer Model

Polyethylene is modeled as a graph `G = (V, E)`:

*   **Nodes (V):** Monomer units
*   **Edges (E):** Covalent bonds
*   Placed in a 3D periodic box simulating an amorphous material

### ⏳ Degradation Dynamics

Simulates changes over days, driven by chemical kinetics:

#### Crosslinking (P_XL):

```math
P_{XL}(t, D) = -a_{XL}(t) \cdot D^{b_{XL}(t)}
```

Where:

*   `a_XL(t) = (7.324e-4)t - 1.034e-3`
*   `b_XL(t) = (-5.631e-5)t + 1.015`

#### Scission (P_SC):

```math
P_{SC}(t, D) = a_{SC}(t) \cdot D^{b_{SC}(t)}
```

Where:

*   `a_SC(t) = (3.385e-4)t^2 + (3.152e-2)t - 0.4905`
*   `b_SC(t) = (1.575e-4)t + 0.5168`

### 🌌 Spatial & Energy Considerations

*   **Spatially-Aware Events**: k-d tree used to prefer nearby bonds during crosslinking.
*   **Energy-Based Scission**: Nodes with more stress (higher degree) have higher break probability.

### 🧮 Measuring Integrity with Graph Laplacian

The integrity of the network is approximated by the algebraic connectivity:

```math
\lambda_2 = \text{Second-smallest eigenvalue of } L = D_{mat} - A
```

Where:

*   `D_mat`: Degree matrix
*   `A`: Adjacency matrix
*   `λ2`: Higher means more connected, stronger material

---

## 🤝 Contributing

We welcome contributions to the Intelligent Polymer Degradation Simulator! Here's how you can help:

### Reporting Bugs

If you find a bug, please open an issue on the [GitHub repository](https://github.com/Cool-Coder174/polyethylene_simulation/issues). Provide a clear and concise description of the bug, steps to reproduce it, and your environment details.

### Suggesting Enhancements

Had an idea for a new feature or an improvement? Open an issue on the [GitHub repository](https://github.com/Cool-Coder174/polyethylene_simulation/issues) and describe your suggestion.

### Making Code Contributions

1.  **Fork the repository:** Click the "Fork" button on the top right of the [GitHub repository](https://github.com/Cool-Coder174/polyethylene_simulation).
2.  **Clone your forked repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/polyethylene_simulation.git
    cd polyethylene_simulation
    ```
3.  **Create a new branch:**
    ```bash
    git checkout -b feature/your-feature-name
    ```
    (or `bugfix/your-bugfix-name` for bug fixes)
4.  **Make your changes:** Implement your feature or bug fix. Ensure your code adheres to the existing style and conventions.
5.  **Write/Update tests:** If you're adding new functionality, please write unit tests. If you're fixing a bug, add a test that reproduces the bug and verifies the fix.
6.  **Run tests and linting:** Before committing, ensure all tests pass and your code is lint-free.
    ```bash
    # (Assuming you have activated your virtual environment)
    python -m pytest src/test_environment.py # Or run all tests if more exist
    # Add linting command here if applicable (e.g., flake8, black, ruff)
    ```
7.  **Commit your changes:** Write a clear and concise commit message.
    ```bash
    git commit -m "feat: Add new feature"
    # or "fix: Resolve bug in X"
    ```
8.  **Push to your fork:**
    ```bash
    git push origin feature/your-feature-name
    ```
9.  **Open a Pull Request:** Go to your forked repository on GitHub and open a pull request to the `main` branch of the original repository. Provide a detailed description of your changes.

---

## 📬 Contact

Maintained by [@Cool-Coder174](https://github.com/Cool-Coder174). Pull requests, issues, and collaborations welcome!

---

## 🧠 Inspiration

Inspired by scientific computing, polymer physics, and reinforcement learning research, this simulator bridges chemistry and AI to accelerate materials science.

---

## 📜 License

[MIT License](LICENSE)

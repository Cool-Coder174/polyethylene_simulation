# 🧪 Intelligent Polymer Degradation Simulator

A deep reinforcement learning (DRL) framework for simulating and optimizing the degradation of polyethylene polymers. This project uses an AI agent to learn the complex physics of polymer chain scission and crosslinking, enabling rapid prediction of long-term material performance under various conditions.

---

## ⚡ What Is This?

Think of it like a virtual scientist. Instead of waiting decades to see how plastic degrades, this project runs fast-forward simulations with an AI that tweaks parameters to see what happens. The AI learns:

*   How polymers break apart or fuse together.
*   What radiation doses or chemical kinetics lead to strong or weak materials.
*   How to find the optimal parameters to match real-world experimental data.

It's like conducting thousands of lab experiments in code to uncover the physics behind polymer aging.

---

## 🚀 Key Features

*   🤖 **AI-Controlled Simulation:** Employs a Soft Actor-Critic (SAC) agent to intelligently tune simulation parameters.
*   🔬 **Hybrid Physics Modeling:** Combines a kinetics-driven ODE model with symbolic regression to discover and refine physical equations.
*   ⚙️ **Automated Hyperparameter Tuning:** Integrated with **Optuna** for efficient optimization of the RL agent.
*   📊 **Robust Data Logging:** Stores all simulation results and metadata in a structured **SQLite** database.
*   🌐 **Interactive Visualization:** Generates interactive plots with **Plotly** to explore simulation outcomes.
*   HPC **Ready:** Includes scripts and guides for running computationally intensive jobs on a SLURM-based cluster.

---

## 🛠 Getting Started

### Prerequisites

*   Python 3.8+
*   A virtual environment manager (`venv` or `conda`)
*   Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Cool-Coder174/polyethylene_simulation.git
    cd polyethylene_simulation
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Using venv
    python3 -m venv venv
    source venv/bin/activate

    # Or using conda
    conda create -n polymer_sim python=3.9
    conda activate polymer_sim
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    Or use the provided shell script:
    ```bash
    bash install_dependencies.sh
    ```

---
## 📖 Usage

### Configuration

*   **`config.yaml`**: The main configuration file. Adjust run parameters, RL hyperparameters, and output paths here.
*   **`kinetic_params.yaml`**: Defines the kinetic rate constants for the chemical ODE model.

### Training the Agent

The primary script for training the agent is `src/train.py`. You can run it in two modes, controlled by the `enable_optuna` flag in `config.yaml`.

*   **Single Run (Optuna Disabled):**
    To run a single training session with the hyperparameters defined in `config.yaml`:
    ```bash
    python src/train.py
    ```

*   **Hyperparameter Optimization (Optuna Enabled):**
    To launch an Optuna study that searches for the best hyperparameters:
    ```bash
    python src/train.py
    ```
    The study will run for `optuna_trials` as specified in the config file.

### Visualizing Results

After running a simulation, you can generate interactive plots from the data stored in the database.

```bash
python src/interactive_plotting.py
```
The plots will be saved in the directory specified by `plot_path` in `config.yaml`.

---

## 🧠 Hybrid Symbolic-RL Modeling

This project implements a novel hybrid modeling approach that combines symbolic regression with deep reinforcement learning to create a more accurate and physically realistic model of polyethylene degradation.

### Workflow

1.  **Symbolic Regression for Scission Rate (`scripts/discover_scission_model.py`)**:
    *   The workflow begins by analyzing experimental data for polymer chain scission.
    *   It uses the `pysr` library to perform symbolic regression, searching for a simple mathematical formula that describes the rate of chain scission.
    *   The best-fit equation is saved and integrated into the main ODE model in `src/polymer_env.py`.

2.  **Reinforcement Learning for Parameter Tuning (`src/train.py`)**:
    *   A Soft Actor-Critic (SAC) agent is trained to fine-tune two key parameters of the simulation:
        1.  A multiplier for the overall crosslinking rate.
        2.  A multiplier for the newly discovered symbolic scission rate.
    *   The agent's goal is to find multipliers that make the simulation output match experimental data for both high and low dose rates simultaneously.

### Reinforcement Learning Loop

*   **State (S)** — 2D vector: `[crosslink_multiplier, scission_multiplier]`
    The state represents the current multipliers applied to the kinetic rates.

*   **Action (A)** — 2D vector: `[Δ_crosslink_multiplier, Δ_scission_multiplier]`
    The agent suggests a multiplicative change to the current multipliers.

*   **Reward (R)**: The reward is the negative mean squared error between the logarithm of the predicted and true scission-to-crosslink ratios, calculated over high and low dose-rate simulations.
    ```math
    R = - \frac{1}{N} \sum_{i=1}^{N} \left( \log\left(\frac{S_{pred}}{C_{pred}}\right)_i - \log\left(\frac{S_{true}}{C_{true}}\right)_i \right)^2
    ```

*   **Episode Termination**: Each episode consists of a single step. The agent proposes a set of multipliers, the environment runs the full simulation for both dose rates, calculates the reward, and terminates.

---

## 💻 Running on an HPC (Kamiak)

For large-scale experiments, it is recommended to run the simulation on a High-Performance Computing (HPC) cluster.

1.  **Configure the Job:** Modify the `submit_kamiak_job.sh` script to set your desired resources (time, memory, etc.).

2.  **Submit the Job:**
    ```bash
    sbatch submit_kamiak_job.sh
    ```
    This script handles loading the necessary modules, activating the environment, and running `src/train.py`.

3.  **Monitor the Job:** Check the status of your submitted job using its ID.
    ```bash
    squeue -u your_username
    ```

For a more detailed guide on using the Kamiak cluster, see `KamiakGuide.md`.

---

## 📂 Project Structure

```
/
├── main.py                     # Orchestrator script for the full workflow
├── config.yaml                 # Main configuration file
├── kinetic_params.yaml         # Kinetic parameters for the ODE model
├── requirements.txt            # Python package dependencies
├── install_dependencies.sh     # Installation script
├── submit_kamiak_job.sh        # SLURM job submission script
├── KamiakGuide.md              # User guide for the Kamiak HPC
├── ReadMe.md                   # This file
│
├── data/
│   └── polyethylene_chain.pdb  # Initial polymer structure data
│
├── models/                     # Directory for saved model artifacts
│
├── results/                    # Default output directory for DB, plots, and models
│
├── src/
│   ├── train.py                # Main training script (RL tuning)
│   ├── polymer_env.py          # Gym-like simulation environment
│   ├── sac_agent.py            # Soft Actor-Critic agent implementation
│   ├── replay_buffer.py        # Replay buffer for the agent
��   ├── database.py             # SQLite database handling
│   └── interactive_plotting.py # Script for generating result plots
│
├── scripts/
│   ├── discover_scission_model.py # Symbolic regression for scission rate
│   └── ...                     # Other utility and setup scripts
│
└── tests/
    └── ...                     # Unit and integration tests
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature`).
3.  Commit your changes (`git commit -m 'feat: Add some feature'`).
4.  Push to the branch (`git push origin feature/your-feature`).
5.  Open a pull request.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
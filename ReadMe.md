# 🧪 Intelligent Polymer Degradation Simulator

A deep reinforcement learning (DRL) framework for simulating and optimizing the degradation of polyethylene polymers. This project uses an AI agent to learn the complex physics of polymer chain scission and crosslinking, enabling rapid prediction of long-term material performance under various conditions.

---

## 🚀 Key Features

*   🤖 **AI-Controlled Simulation:** Employs a Soft Actor-Critic (SAC) agent to intelligently tune simulation parameters.
*   🔬 **Hybrid Physics Modeling:** Combines a kinetics-driven ODE model with symbolic regression to discover and refine physical equations.
*   ⚙️ **Automated Hyperparameter Tuning:** Integrated with **Optuna** for efficient optimization of the RL agent.
*   📊 **Robust Data Logging:** Stores all simulation results and metadata in a structured **SQLite** database.
*   🌐 **Interactive Visualization:** Generates interactive plots with **Plotly** to explore simulation outcomes.
*    HPC **Ready:** Includes scripts and guides for running computationally intensive jobs on a SLURM-based cluster like Kamiak.

---

## Workflow

The project follows a two-stage hybrid modeling approach:

1.  **Symbolic Model Discovery (Optional):**
    The `scripts/discover_scission_model.py` script uses symbolic regression (`pysr`) to find a mathematical equation for the polymer chain scission rate from experimental data. The resulting equation is saved and used in the simulation environment.

2.  **Reinforcement Learning for Parameter Tuning:**
    The main training script, `src/train.py`, uses a SAC agent to fine-tune the parameters of the physics simulation. The agent's goal is to adjust multipliers for the scission and crosslinking rates until the simulation's output matches ground-truth experimental data for both high and low radiation dose rates.

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
    The `install_dependencies.sh` script installs all required packages.
    ```bash
    bash install_dependencies.sh
    ```
    Alternatively, you can install them directly using pip:
    ```bash
    pip install -r requirements.txt
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

## 💻 Running on an HPC (Kamiak)

For large-scale experiments, it is recommended to run the simulation on a High-Performance Computing (HPC) cluster.

1.  **Configure the Job:** Modify the `submit_kamiak_job.sh` script to set your desired resources (time, memory, etc.).
2.  **Submit the Job:**
    ```bash
    sbatch submit_kamiak_job.sh
    ```
    This script handles loading the necessary modules, activating the environment, and running `src/train.py`.

For a detailed guide on using the Kamiak cluster, see `KamiakGuide.md`.

---

## 📂 Project Structure

```
/
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
│   ├── train.py                # Main training script (entry point)
│   ├── polymer_env.py          # Gym-like simulation environment
│   ├── sac_agent.py            # Soft Actor-Critic agent implementation
│   ├── replay_buffer.py        # Replay buffer for the agent
│   ├── database.py             # SQLite database handling
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

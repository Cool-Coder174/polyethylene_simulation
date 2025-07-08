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
*   `data/` — Contains initial polymer chain data (`polyethylene_chain.pdb`).
*   `models/` — Directory for saving trained agent models and discovered symbolic equations.
*   `results/` — Output directory for simulation data, plots, and saved models.
*   `scripts/` — Contains helper and discovery scripts.
    *   `discover_scission_model.py` — Uses symbolic regression (PySR) to find a scission rate equation.
*   `src/` — Contains all Python source code for the simulation and RL agent.
    *   `fine_tune_model.py` — **Main entry point.** Orchestrates the entire workflow.
    *   `train.py` — Legacy training script with Optuna support.
    *   `polymer_env.py` — Gym-compatible physics simulation environment.
    *   `sac_agent.py` — Implementation of the Distributional Soft Actor-Critic (DSAC) algorithm.
    *   `replay_buffer.py` — Manages the experience replay buffer.
    *   `database.py` — Handles SQLite database interactions.
    *   `interactive_plotting.py` — Generates interactive visualizations.
*   `submit_kamiak_job.sh` — Script for submitting the fine-tuning job to a SLURM-based HPC.

### 🧠 Reinforcement Learning Loop (Parameter Tuning)

The RL loop is designed to fine-tune two key parameters of the kinetic model: a multiplier for the crosslinking rate and a multiplier for the scission rate.

*   **State (S)** — 2D vector:
    `[crosslink_multiplier, scission_multiplier]`

*   **Action (A)** — 2D vector:
    `[Δ_crosslink_multiplier, Δ_scission_multiplier]`
    (The agent suggests a multiplicative change to the current multipliers.)

*   **Reward (R)**: The reward is the negative mean squared error between the logarithm of the predicted and true scission-to-crosslink ratios, calculated over high and low dose-rate simulations.

    ```math
R = - \frac{1}{N} \sum_{i=1}^{N} \left( \log\left(\frac{S_{pred}}{C_{pred}}\right)_i - \log\left(\frac{S_{true}}{C_{true}}\right)_i \right)^2
    ```

*   **Episode Termination**: Each episode consists of a single step. The agent proposes a set of multipliers, the environment runs the full simulation for both dose rates, calculates the reward, and terminates.

---

## 🔬 Hybrid Symbolic-RL Modeling

This project implements a novel hybrid modeling approach that combines symbolic regression with deep reinforcement learning to create a more accurate and physically realistic model of polyethylene degradation.

### Workflow

1.  **Symbolic Regression for Scission Rate (`scripts/discover_scission_model.py`)**:
    *   The workflow begins by analyzing experimental data for polymer chain scission.
    *   It uses the `pysr` library to perform symbolic regression, searching for a simple mathematical formula that describes the rate of chain scission as a function of radiation dose rate and time.
    *   The best-fit equation is saved to `models/scission_equation.json`.

2.  **Reinforcement Learning for Parameter Tuning (`fine_tune_model.py`)**:
    *   The symbolic scission equation discovered in the first step is integrated into the main ODE model in `src/polymer_env.py`.
    *   A Distributional Soft Actor-Critic (DSAC) agent is then trained to fine-tune two key parameters:
        1.  A multiplier for the overall crosslinking rate.
        2.  A multiplier for the newly discovered symbolic scission rate.
    *   The agent's goal is to find multipliers that make the simulation output match experimental data for both high and low dose rates simultaneously.

### Running the Full Workflow on an HPC (Kamiak)

The combined symbolic regression and RL tuning process is computationally intensive and is best run on a High-Performance Computing (HPC) cluster.

1.  **Setup your Conda Environment**: Ensure you have a Conda environment with all the necessary packages from `requirements.txt` installed. Activate it.

2.  **Submit the Job**: Use the provided SLURM script to submit the job.
    ```bash
sbatch submit_kamiak_job.sh
    ```
    This script will:
    *   Request the necessary compute resources (CPUs, memory, time).
    *   Load the Anaconda module and activate your environment.
    *   Run the main `fine_tune_model.py` script.
    *   Save SLURM output and error logs to the `slurm_output/` directory.

3.  **Monitor the Job**: Check the status of your job using:
    ```bash
squeue -u your_username
    ```

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
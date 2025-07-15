## 🚀 A User‑Friendly Guide to the WSU Kamiak Supercomputer

This guide provides instructions for running the polyethylene simulation project on the WSU Kamiak Supercomputer.

---

### 1. Connecting to Kamiak

Connect to Kamiak using SSH:

```bash
ssh your.wsu.id@kamiak.wsu.edu
```

---

### 2. Setting up the Environment

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Cool-Coder174/polyethylene_simulation.git
    cd polyethylene_simulation
    ```

2.  **Load required modules:**
    ```bash
    module load python/3.9.1 lammps
    ```

3.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

### 3. Running Jobs

#### Interactive Jobs

For quick tests and debugging, use an interactive job:

```bash
idev --time=01:00:00 --cpus-per-task=8
```

Once on a compute node, you can run the scripts directly.

#### Batch Jobs

For longer runs, use a batch script. A sample script, `submit_kamiak_job.sh`, is provided in the repository. Modify it to set your desired resources and then submit it:

```bash
sbatch submit_kamiak_job.sh
```

The script will run the `src/train.py` script, which trains the RL agent.

---

### 4. Monitoring Jobs

*   **Check job status:**
    ```bash
    squeue -u your.wsu.id
    ```
*   **View job history:**
    ```bash
    sacct -u your.wsu.id
    ```
*   **Cancel a job:**
    ```bash
    scancel JOBID
    ```

---

### 5. Sample Batch Script (`submit_kamiak_job.sh`)

```bash
#!/bin/bash
#SBATCH --job-name=pe_simulation
#SBATCH --partition=kamiak
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.wsu.id@wsu.edu

# Load modules
module load python/3.9.1 lammps

# Activate environment
source venv/bin/activate

# Run training script
python src/train.py
```


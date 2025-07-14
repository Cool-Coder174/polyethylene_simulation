#!/bin/bash

#SBATCH --partition=kamiak       # Queue/Partition to run in
#SBATCH --job-name=Polymer-Hybrid-Main # A descriptive name for your job
#SBATCH --output=slurm_output/%x_%j.out # Standard output log
#SBATCH --error=slurm_output/%x_%j.err  # Standard error log
#SBATCH --nodes=1                # We need all cores on a single node
#SBATCH --ntasks-per-node=1      # Run a single task
#SBATCH --cpus-per-task=32       # Request 32 cores for that task
#SBATCH --mem=128G               # Request 128 Gigabytes of memory
#SBATCH --time=2-00:00:00        # Wall clock time limit: 2 days
#SBATCH --gres=gpu:1             # Request 1 GPU

# --- Job Setup ---
# Create directory for SLURM logs if it doesn't exist
mkdir -p slurm_output

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOBID"

# --- Environment Setup ---
# Load necessary modules. Use Anaconda for managing Python dependencies.
module load anaconda3
module load cuda/12.2.0 # Load CUDA module for GPU support
module load julia/1.11.4      # Load Julia module for PySR

# Initialize Conda for shell interaction
eval "$(conda shell.bash hook)"

# Replace 'your_conda_env_name' with the name of your actual Conda environment
conda activate polyethylene_simulation 

# --- Run the main script ---
# The srun command executes the script on the allocated compute node
echo "Running the main workflow script..."
srun python main.py

echo "Job finished at $(date)"

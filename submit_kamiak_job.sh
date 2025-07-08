#!/bin/bash

#SBATCH --partition=kamiak       # Queue/Partition to run in
#SBATCH --job-name=Polymer-Hybrid  # A descriptive name for your job
#SBATCH --output=slurm_output/%x_%j.out # Standard output log
#SBATCH --error=slurm_output/%x_%j.err  # Standard error log
#SBATCH --nodes=1                # We need all cores on a single node
#SBATCH --ntasks-per-node=1      # Run a single task
#SBATCH --cpus-per-task=16       # Request 16 cores for that task
#SBATCH --mem=64G                # Request 64 Gigabytes of memory
#SBATCH --time=2-00:00:00        # Wall clock time limit: 2 days

# --- Job Setup ---
# Create directory for SLURM logs if it doesn't exist
mkdir -p slurm_output

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOBID"

# --- Environment Setup ---
# Load necessary modules. Use Anaconda for managing Python dependencies.
module load anaconda3

# Initialize Conda for shell interaction
eval "$(conda shell.bash hook)"

# Replace 'your_conda_env_name' with the name of your actual Conda environment
conda activate polyethylene_simulation 

# --- Run the main script ---
# The srun command executes the script on the allocated compute node
echo "Running the fine-tuning script..."
srun python fine_tune_model.py

echo "Job finished at $(date)"

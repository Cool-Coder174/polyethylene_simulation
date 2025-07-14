#!/usr/bin/env bash
#SBATCH --job-name=polyethylene_test
#SBATCH --partition=kamiak
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:tesla:1
#SBATCH --output=tests_out/test_results_%j.out
#SBATCH --error=tests_out/test_errors_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=i.hernandez-domingu@wsu.edu

echo "Starting test run at $(date)"

# Activate existing Conda environment
conda activate polyethylene_simulation

# Run tests
echo "Running pytest..."
python -m pytest tests/

echo "Test run finished at $(date)"

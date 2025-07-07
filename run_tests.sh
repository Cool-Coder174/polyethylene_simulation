#!/bin/bash
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
#SBATCH --mail-user=i.hernandez-domingu@wsu.edu # Please replace with your email

echo "Starting test run at $(date)"

# Load modules
module purge
module load StdEnv
module load python3/3.9.5 cuda/12.2.0

# Create and activate virtual environment
VENV_DIR=~/venvs/polyethylene
python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run tests
echo "Running pytest..."
python -m pytest tests/


echo "Test run finished at $(date)"

#!/usr/bin/env bash
#SBATCH --job-name=polyethylene_test
#SBATCH --partition=kamiak
#SBATCH --time=02:55:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:tesla:1
#SBATCH --output=tests_out/test_results_%j.out
#SBATCH --error=tests_out/test_errors_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=i.hernandez-domingu@wsu.edu

echo "Starting test run at $(date)"

# Load modules
module purge
module load StdEnv
module load python3/3.11.4 cuda/12.2.0
export LD_LIBRARY_PATH=$HOME/plugins/libffi:$LD_LIBRARY_PATH
export CPATH=$HOME/plugins/libffi/include:$CPATH
export LD_LIBRARY_PATH=/opt/apps/cuda/12.2.0/gcc/6.1.0/nvvm/lib64:$LD_LIBRARY_PATH

# Create and activate virtual environment
VENV_DIR=~/venvs/polyethylene
python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run tests
echo "Running pytest..."
$VENV_DIR/bin/python -m pytest tests/


echo "Test run finished at $(date)"

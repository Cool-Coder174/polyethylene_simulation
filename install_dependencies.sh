#!/bin/bash

# This script automates the installation of necessary dependencies for the
# Polyethylene Degradation Simulation project. It detects the operating system
# and installs Python packages listed in `requirements.txt`.
#
# Before running this script, ensure you have Git installed to clone the repository
# and a recent version of Python 3 (3.8+) with `pip` available on your system.
#
# Usage:
#   Navigate to the project's root directory in your terminal and run:
#   bash install_dependencies.sh

# --- 1. Detect Operating System ---
# This section identifies the operating system to provide platform-specific advice.
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    CYGWIN*)    MACHINE=Cygwin;;
    MINGW*)     MACHINE=MinGw;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

echo "Detected Operating System: ${MACHINE}"

# --- 2. Check for Python 3 and pip ---
# It's crucial to have Python 3 and its package installer, pip, installed.
# This script will check for their presence and guide the user if they are missing.

# Check if 'python3' command is available.
if ! command -v python3 &> /dev/null
then
    echo "\nError: 'python3' command not found."
    echo "Please install Python 3 (version 3.8 or higher) and ensure it's in your system's PATH."
    echo "  - For Ubuntu/Debian: sudo apt update && sudo apt install python3 python3-pip python3-venv"
    echo "  - For macOS: Install Homebrew (brew.sh), then: brew install python"
    echo "  - For Windows: Download from python.org and ensure 'Add Python to PATH' is checked during installation."
    exit 1
fi

# Check if 'pip3' command is available.
if ! command -v pip3 &> /dev/null
then
    echo "\nError: 'pip3' command not found."
    echo "Pip is usually installed with Python 3. If not, you might need to install it separately."
    echo "  - Try: python3 -m ensurepip --default-pip"
    echo "  - Or refer to your OS-specific Python/pip installation guide."
    exit 1
fi

echo "Python 3 and pip3 found. Proceeding with dependency installation."

# --- 3. Check for Julia ---
# PySR, a dependency for symbolic regression, requires Julia to be installed.
if ! command -v julia &> /dev/null
then
    echo "\nWarning: 'julia' command not found."
    echo "PySR requires Julia. Please install it from: https://julialang.org/downloads/"
    echo "For Linux and macOS, the recommended way is to use juliaup:"
    echo "curl -fsSL https://install.julialang.org | sh"
fi

# --- 4. Install Python Dependencies from requirements.txt ---
# This step installs all required Python libraries using pip.
# The dependencies are listed in the 'requirements.txt' file.
echo "\nInstalling Python dependencies from requirements.txt..."
pip3 install -r requirements.txt

# Check the exit status of the last command (pip3 install).
if [ $? -eq 0 ]; then
    echo "Python dependencies installed successfully."
else
    echo "\nError: Failed to install Python dependencies."
    echo "Please review the error messages above. Common issues include network problems,"
    echo "missing build tools (e.g., for scipy/numpy), or incompatible Python versions."
    exit 1
fi

# --- 5. Platform-Specific Notes and Additional Considerations ---
# Provides guidance for specific operating systems regarding common prerequisites
# like SQLite and OpenMM with CUDA support.
echo "\n--- Additional Setup Notes ---"

if [ "${MACHINE}" == "Linux" ]; then
    echo "  - SQLite: Usually pre-installed. If not, install via your package manager (e.g., 'sudo apt install sqlite3' on Debian/Ubuntu)."
    echo "  - OpenMM with CUDA: Requires NVIDIA drivers and CUDA Toolkit. OpenMM's pip package often includes CUDA binaries, but verify your setup. Consult OpenMM documentation for advanced GPU configurations."
elif [ "${MACHINE}" == "Mac" ]; then
    echo "  - SQLite: Usually pre-installed. If not, install via Homebrew ('brew install sqlite')."
    echo "  - OpenMM with CUDA: Requires NVIDIA drivers and CUDA Toolkit. Ensure compatibility with your macOS version. Consult OpenMM documentation for details."
elif [ "${MACHINE}" == "MinGw" ] || [ "${MACHINE}" == "Cygwin" ]; then
    echo "  - Windows: It's recommended to use Git Bash or Windows Subsystem for Linux (WSL) for running this script."
    echo "  - SQLite: Can be downloaded from sqlite.org or often comes with Python installations."
    echo "  - OpenMM with CUDA: Requires NVIDIA drivers and CUDA Toolkit. Ensure your environment variables are correctly set up. Consult OpenMM documentation for details."
else
    echo "  - Unknown operating system. Please refer to the project README for manual dependency installation instructions."
fi

echo "\nSetup script finished. Please review any messages above for further actions or issues."
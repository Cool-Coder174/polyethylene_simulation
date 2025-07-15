"""
This script runs a reactive molecular dynamics simulation using LAMMPS.
"""
import subprocess
from pathlib import Path

def run_reactive_md():
    """
    Main function to orchestrate the reactive MD simulation with LAMMPS.
    """
    print("--- Running Reactive MD with LAMMPS ---")
    
    # --- 1. Define Paths ---
    lammps_in_path = Path("data/pe_system.in")
    lammps_log_path = Path("data/pe_system.log")

    if not lammps_in_path.exists():
        print(f"Error: {lammps_in_path} not found. Please run setup_md_system.py first.")
        return

    # --- 2. Run LAMMPS Simulation ---
    # The `lammps` executable must be in the system's PATH.
    command = f"lammps -in {lammps_in_path} -log {lammps_log_path}"
    print(f"Executing command: {command}")
    
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"LAMMPS simulation completed successfully. Log file at: {lammps_log_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running LAMMPS: {e}")
        return

    # --- 3. (Optional) Parse LAMMPS Output ---
    # Here you could add code to parse the log file or other output files
    # to extract quantities of interest, like the number of crosslinks or scissions.
    # For this example, we will just confirm the simulation ran.

if __name__ == "__main__":
    run_reactive_md()

"""
This script runs a reactive molecular dynamics simulation using LAMMPS and parses the output.
"""
import subprocess
from pathlib import Path
import re
import json
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_lammps_log(log_path: Path) -> dict:
    """
    Parses the LAMMPS log file to extract simulation data.

    Args:
        log_path (Path): The path to the LAMMPS log file.

    Returns:
        dict: A dictionary containing the parsed data.
    """
    data = {
        "timesteps": [],
        "temp": [],
        "press": [],
        "toteng": [],
        "f_rx": [] # Output from fix rx
    }

    with open(log_path, 'r') as f:
        log_content = f.read()

    # Find the start and end of the thermo data
    start_match = re.search(r"Step\s+Temp\s+Press\s+TotEng\s+f_rx", log_content)
    end_match = re.search(r"Loop time of", log_content)

    if not start_match or not end_match:
        return {}

    data_block = log_content[start_match.end():end_match.start()]

    for line in data_block.strip().split('\n'):
        parts = line.split()
        if len(parts) == 5: # Now expecting 5 columns: Step, Temp, Press, TotEng, f_rx
            data["timesteps"].append(int(parts[0]))
            data["temp"].append(float(parts[1]))
            data["press"].append(float(parts[2]))
            data["toteng"].append(float(parts[3]))
            data["f_rx"].append(float(parts[4])) # Assuming f_rx is a float or int

    return data

def run_reactive_md():
    """
    Main function to orchestrate the reactive MD simulation with LAMMPS.
    """
    logging.info("--- Running Reactive MD with LAMMPS ---")
    
    # Load configuration
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- 1. Define Paths from config ---
    lammps_params = config['lammps_parameters']
    lammps_in_path = Path(lammps_params['input_script_path'])
    lammps_log_path = Path(lammps_params['log_path'])
    lammps_executable = lammps_params['executable_path']

    if not lammps_in_path.exists():
        logging.error(f"Error: {lammps_in_path} not found. Please run setup_md_system.py first.")
        return

    # --- 2. Run LAMMPS Simulation ---
    command = f"{lammps_executable} -in {lammps_in_path} -log {lammps_log_path}"
    logging.info(f"Executing command: {command}")
    
    try:
        subprocess.run(command, shell=True, check=True)
        logging.info(f"LAMMPS simulation completed successfully. Log file at: {lammps_log_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running LAMMPS: {e}")
        return

    # --- 3. Parse LAMMPS Output ---
    parsed_data = parse_lammps_log(lammps_log_path)
    if not parsed_data:
        logging.error("Could not parse LAMMPS log file.")
        return

    # --- 4. Save Parsed Data ---
    output_path = Path(config['output_path']) # Load output path from config
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(parsed_data, f, indent=4)
        
    logging.info(f"Parsed LAMMPS data saved to {output_path}")

if __name__ == "__main__":
    run_reactive_md()

"""
Main entry point for the entire polyethylene simulation and modeling workflow.

This script orchestrates the following steps:
1.  Checks for necessary external dependencies (like Julia).
2.  Runs the molecular dynamics (MD) system setup.
3.  Runs the symbolic regression script to discover the scission rate model.
4.  Runs the reactive MD script to calculate ab-initio rate constants.
5.  Runs the main reinforcement learning fine-tuning process.
"""
import subprocess
import sys
import shutil
import logging
import argparse
import yaml
from pathlib import Path

# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def load_config(config_path="config.yaml"):
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        sys.exit(1)

def print_header(message):
    """Prints a formatted header to the log."""
    logging.info("="*70)
    logging.info(f"▶ {message}")
    logging.info("="*70)

def check_dependencies():
    """Checks for essential external dependencies."""
    print_header("Step 1: Checking Dependencies")
    if not shutil.which("julia"):
        logging.error(
            "Julia is not found in your system's PATH. "
            "It is required for symbolic regression with PySR. "
            "Please install it from: https://julialang.org/downloads/"
        )
        sys.exit(1)
    logging.info("✅ SUCCESS: All essential dependencies are found.")

def run_script(script_path: str, description: str, capture_output: bool = False):
    """Helper function to run an external Python script."""
    logging.info(f"Running: {description}")
    script_path = Path(script_path)
    if not script_path.exists():
        logging.error(f"Script not found: {script_path}")
        return False

    try:
        # Use a pipe to stream output in real-time if not capturing
        if not capture_output:
            with subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            ) as proc:
                for line in proc.stdout:
                    logging.info(line.strip())
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, proc.args)
        else:
            # Capture output for quieter steps
            result = subprocess.run(
                [sys.executable, str(script_path)],
                check=True, capture_output=True, text=True
            )
            if result.stdout:
                logging.info(f"--- STDOUT ---\n{result.stdout.strip()}")
            if result.stderr:
                logging.warning(f"--- STDERR ---\n{result.stderr.strip()}")

        logging.info(f"✅ SUCCESS: {description} complete.")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ ERROR: Failed during '{description}'.")
        if hasattr(e, 'stdout') and e.stdout:
            logging.error(f"--- STDOUT ---\n{e.stdout.strip()}")
        if hasattr(e, 'stderr') and e.stderr:
            logging.error(f"--- STDERR ---\n{e.stderr.strip()}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during '{description}': {e}")
        return False


def main():
    """Main function to parse arguments and run the workflow."""
    parser = argparse.ArgumentParser(description="Polyethylene Simulation Workflow Orchestrator")
    parser.add_argument(
        "--steps",
        nargs='+',
        type=int,
        default=list(range(1, 6)),
        help="A list of steps to run (e.g., --steps 1 3 5). Default is all steps."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file."
    )
    args = parser.parse_args()
    
    config = load_config(args.config)
    paths = config.get("script_paths", {})

    # --- Workflow Steps ---
    workflow_steps = {
        1: ("Dependencies Check", check_dependencies, {}),
        2: ("MD System Setup", run_script, {"script_path": paths.get("md_setup"), "description": "MD System Setup", "capture_output": True}),
        3: ("Symbolic Regression", run_script, {"script_path": paths.get("symbolic_regression"), "description": "Symbolic Regression"}),
        4: ("Reactive MD", run_script, {"script_path": paths.get("reactive_md"), "description": "Reactive MD"}),
        5: ("RL Fine-Tuning", run_script, {"script_path": paths.get("rl_fine_tuning"), "description": "RL Fine-Tuning"}),
    }

    for step_num in sorted(args.steps):
        if step_num in workflow_steps:
            title, func, kwargs = workflow_steps[step_num]
            print_header(f"Executing Step {step_num}: {title}")
            
            # Special handling for dependency check which is not a script
            if step_num == 1:
                func()
                continue

            if not func(**kwargs):
                logging.error(f"Step {step_num} failed. Aborting workflow.")
                sys.exit(1)
        else:
            logging.warning(f"Step {step_num} is not a valid step. Skipping.")

    print_header("Workflow Complete!")
    logging.info("✅ All selected tasks finished successfully!")

if __name__ == "__main__":
    main()

Main entry point for the entire polyethylene simulation and modeling workflow.

This script orchestrates the following steps in order:
1.  Checks for necessary external dependencies (like Julia).
2.  Runs the molecular dynamics (MD) system setup to create an equilibrated polymer cell.
3.  Runs the symbolic regression script to discover the scission rate model.
4.  Runs the reactive MD script to calculate ab-initio rate constants (with placeholders).
5.  Runs the main reinforcement learning fine-tuning process.


import subprocess
import sys
import shutil
from pathlib import Path

def print_header(message):
    """Prints a formatted header."""
    print("\n" + "="*70)
    print(f"▶ {message}")
    print("="*70)

def print_success(message):
    """Prints a success message."""
    print(f"✅ SUCCESS: {message}")

def print_warning(message):
    """Prints a warning message."""
    print(f"⚠️ WARNING: {message}")

def print_error(message):
    """Prints an error message and exits."""
    print(f"❌ ERROR: {message}", file=sys.stderr)
    sys.exit(1)

def check_dependencies():
    """Checks for essential external dependencies."""
    print_header("Step 1: Checking Dependencies")
    
    # Check for Julia, which is required by PySR
    if not shutil.which("julia"):
        print_error(
            "Julia is not found in your system's PATH.\n"
            "Julia is required for symbolic regression with PySR.\n"
            "Please install it from: https://julialang.org/downloads/"
        )
    
    print_success("All essential dependencies are found.")

def run_md_setup():
    """Runs the OpenMM system setup script."""
    print_header("Step 2: Setting up Molecular Dynamics System")
    try:
        subprocess.run([sys.executable, "scripts/setup_md_system.py"], check=True, capture_output=True, text=True)
        print_success("MD system setup complete. Equilibrated system saved in 'data/'.")
    except subprocess.CalledProcessError as e:
        print_error(
            f"Failed to set up the MD system.\n"
            f"This could be due to an issue with your OpenMM installation or missing PDB files.\n"
            f"Please check the logs for more details.\n\n"
            f"--- STDOUT ---\n{e.stdout}\n"
            f"--- STDERR ---\n{e.stderr}"
        )

def run_symbolic_regression():
    """Runs the PySR symbolic regression script."""
    print_header("Step 3: Discovering Scission Model via Symbolic Regression")
    print("This step may take a significant amount of time...")
    try:
        # First, ensure PySR's Julia environment is set up.
        # We suppress output unless there's an error.
        subprocess.run(
            [sys.executable, "-c", "import pysr; pysr.install()"], 
            check=True, capture_output=True, text=True
        )
        # Now run the discovery script
        subprocess.run(
            [sys.executable, "scripts/discover_scission_model.py"], 
            check=True, text=True
        )
        print_success("Symbolic regression complete. Scission model saved to 'models/scission_equation.json'.")
    except subprocess.CalledProcessError as e:
        print_error(
            f"Symbolic regression failed.\n"
            f"This could be due to an issue with your PySR or Julia installation.\n"
            f"Please check the logs for more details.\n\n"
            f"--- STDOUT ---\n{e.stdout}\n"
            f"--- STDERR ---\n{e.stderr}"
        )

def run_reactive_md():
    """Runs the reactive MD script to calculate rate constants."""
    print_header("Step 4: Calculating Ab-Initio Rate Constants")
    print_warning(
        "The reactive MD simulation uses PLACEHOLDER values for activation energies.\n"
        "For a real scientific study, you must replace the placeholder function in\n"
        "'scripts/run_reactive_md.py' with an actual QM/MM engine interface."
    )
    try:
        subprocess.run([sys.executable, "scripts/run_reactive_md.py"], check=True, text=True)
        print_success("Reactive MD simulation complete. Ab-initio parameters saved to 'models/ab_initio_params.json'.")
    except subprocess.CalledProcessError as e:
        print_error(
            f"Reactive MD script failed.\n"
            f"Please ensure the equilibrated system from Step 2 exists.\n\n"
            f"--- STDOUT ---\n{e.stdout}\n"
            f"--- STDERR ---\n{e.stderr}"
        )

def run_rl_fine_tuning():
    """Runs the main DSAC fine-tuning script."""
    print_header("Step 5: Fine-Tuning Model with Reinforcement Learning")
    print_warning(
        "The RL fine-tuning process is computationally intensive and may take a very long time.\n"
        "For a full run, it is highly recommended to use the 'submit_kamiak_job.sh' script on an HPC."
    )
    print("Starting RL training...")
    try:
        # Use a pipe to stream output in real-time
        with subprocess.Popen(
            [sys.executable, "fine_tune_model.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        ) as proc:
            for line in proc.stdout:
                print(line, end='')
        
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, proc.args)

        print_success("RL fine-tuning complete. Final parameters saved to 'models/param_multipliers.json'.")
    except subprocess.CalledProcessError as e:
        print_error(
            f"RL fine-tuning failed.\n"
            f"Please check the output above for errors from the training script."
        )

def main():
    """Main function to run the entire workflow."""
    check_dependencies()
    run_md_setup()
    run_symbolic_regression()
    run_reactive_md()
    run_rl_fine_tuning()
    
    print_header("Workflow Complete!")
    print("✅ All tasks complete! Simulation data is ready!")

if __name__ == "__main__":
    main()

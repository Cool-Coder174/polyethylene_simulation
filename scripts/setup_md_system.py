
"""
This script sets up a small, amorphous polyethylene simulation cell using ASE and LAMMPS.
"""
import ase.io
from ase.build import polymer
from ase.io import lammpsdata
from pathlib import Path
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_md_system():
    """
    Builds a polyethylene system and writes LAMMPS input files.
    """
    logging.info("--- Setting up Molecular Dynamics System for LAMMPS ---")

    # Load configuration
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- 1. System Configuration ---
    n_chains = config['run_parameters']['n_chain']
    n_monomers_per_chain = config['run_parameters']['l_chain']
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    output_data = output_dir / "pe_system.data"
    output_in = output_dir / "pe_system.in"

    # --- 2. Build Polymer Topology using ASE ---
    # Create a single polyethylene monomer
    pe_monomer = ase.Atoms('C2H4', positions=[(0, 0, 0), (1.54, 0, 0), 
                                             ( -0.54, 1.0, 0), ( -0.54, -1.0, 0),
                                             (2.08, 1.0, 0), (2.08, -1.0, 0)])
    # Create a polymer chain from the monomer
    pe_polymer = polymer(pe_monomer, a=3.8, n=n_monomers_per_chain)

    # Replicate the polymer to create a simulation box
    pe_polymers = [pe_polymer.copy() for _ in range(n_chains)]
    for i, p in enumerate(pe_polymers):
        p.translate([i * 2, i * 2, 0]) # Simple translation to avoid overlap
    
    system = pe_polymers[0]
    for i in range(1, len(pe_polymers)):
        system.extend(pe_polymers[i])

    system.set_cell([20, 20, 20])
    system.center()

    # --- 3. Write LAMMPS Data File ---
    lammpsdata.write_lammps_data(str(output_data), system, atom_style='full')
    logging.info(f"LAMMPS data file saved to {output_data}")

    # --- 4. Write LAMMPS Input Script ---
    lammps_params = config['lammps_parameters']
    force_field_params = config['lammps_force_field_parameters'] # New section in config.yaml

    # Read reactions.in content
    reactions_path = Path(__file__).parent.parent / 'reactions.in'
    with open(reactions_path, 'r') as f:
        reactions_content = f.read()

    with open(output_in, 'w') as f:
        f.write("# Polyethylene simulation input script\n")
        f.write("units          real\n")
        f.write("atom_style     full\n")
        f.write(f"read_data      {output_data}\n")
        f.write("\n")
        f.write("# Define force field parameters\n")
        f.write(f"pair_style     {force_field_params['pair_style']}\n")
        f.write(f"pair_coeff     {force_field_params['pair_coeff']}\n")
        f.write(f"bond_style     {force_field_params['bond_style']}\n")
        f.write(f"bond_coeff     {force_field_params['bond_coeff']}\n")
        f.write("\n")
        f.write("# Include reactions\n")
        f.write(reactions_content) # Include content of reactions.in directly
        f.write("\n")
        f.write("# Define thermo output\n")
        f.write("thermo_style   custom step temp press toteng f_rx\n")
        f.write("thermo         100\n")
        f.write("\n")
        f.write("# Run simulation\n")
        f.write(f"fix            1 all nvt temp {lammps_params['temp']} {lammps_params['temp']} {lammps_params['temp_damping']}\n")
        f.write(f"timestep       {lammps_params['timestep']}\n")
        f.write(f"run            {lammps_params['run_steps']}\n")

    logging.info(f"LAMMPS input script saved to {output_in}")

if __name__ == "__main__":
    setup_md_system()

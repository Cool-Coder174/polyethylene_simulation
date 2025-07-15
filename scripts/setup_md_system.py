"""
This script sets up a small, amorphous polyethylene simulation cell using ASE and LAMMPS.
"""
import ase.io
from ase.build import polymer
from ase.io import lammpsdata
from pathlib import Path

def setup_md_system():
    """
    Builds a polyethylene system and writes LAMMPS input files.
    """
    print("--- Setting up Molecular Dynamics System for LAMMPS ---")

    # --- 1. System Configuration ---
    n_chains = 10
    n_monomers_per_chain = 50
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    output_data = output_dir / "pe_system.data"
    output_in = output_dir / "pe_system.in"

    # --- 2. Build Polymer Topology using ASE ---
    # Create a single polyethylene chain
    pe_monomer = ase.Atoms('C2H4', positions=[(0, 0, 0), (1.54, 0, 0), 
                                             ( -0.54, 1.0, 0), ( -0.54, -1.0, 0),
                                             (2.08, 1.0, 0), (2.08, -1.0, 0)])
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
    print(f"LAMMPS data file saved to {output_data}")

    # --- 4. Write LAMMPS Input Script ---
    with open(output_in, 'w') as f:
        f.write("units          real\n")
        f.write("atom_style     full\n")
        f.write(f"read_data      {output_data}\n")
        f.write("\n")
        f.write("pair_style     lj/cut 2.5\n")
        f.write("pair_coeff     1 1 1.0 1.0 2.5\n")
        f.write("bond_style     harmonic\n")
        f.write("bond_coeff     1 100.0 1.54\n")
        f.write("\n")
        f.write("neighbor       2.0 bin\n")
        f.write("neigh_modify   delay 10\n")
        f.write("\n")
        f.write("timestep       1.0\n")
        f.write("thermo         100\n")
        f.write("\n")
        f.write("fix            1 all nvt temp 300.0 300.0 100.0\n")
        f.write("run            10000\n")

    print(f"LAMMPS input script saved to {output_in}")

if __name__ == "__main__":
    setup_md_system()
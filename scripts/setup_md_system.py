"""
This script sets up a small, amorphous polyethylene simulation cell using OpenMM,
performs energy minimization and equilibration, and saves the final system state.
"""
import openmm as mm
from openmm import app, unit
from pathlib import Path

def setup_md_system():
    """
    Builds, parameterizes, and equilibrates a polyethylene system.
    """
    print("--- Setting up Molecular Dynamics System ---")

    # --- 1. System Configuration ---
    n_chains = 50
    n_monomers_per_chain = 100
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    output_pdb = output_dir / "equilibrated_pe_system.pdb"
    output_xml = output_dir / "equilibrated_pe_system.xml"

    # --- 2. Build Polymer Topology ---
    # Create a PDB file for a single polyethylene chain (C100H202)
    # This is a simplified representation. For GAFF2, atom types would be crucial.
    # For OPLS-AA, we rely on OpenMM's residue templates.
    pdb = app.PDBFile(str(output_dir / 'polyethylene_chain.pdb'))
    
    # --- 3. Setup Simulation using GAFF Force Field ---
    # GAFF is a good choice for alkanes, and is included in amber14-all.xml
    forcefield = app.ForceField('amber14-all.xml', 'pe.xml')
    
    modeller = app.Modeller(pdb.topology, pdb.positions)
    
    # Add missing hydrogens to the topology
    print("Adding missing hydrogens...")
    modeller.addHydrogens(forcefield)
    
    # Add more chains to create an amorphous cell
    # This is a simplistic packing method. More advanced tools like Packmol
    # would provide better results but require external dependencies.
    for i in range(n_chains - 1):
        modeller.add(pdb.topology, pdb.positions)

    print(f"System contains {modeller.topology.getNumAtoms()} atoms.")

    # --- 4. Create System and Set up Simulation ---
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME,
                                     nonbondedCutoff=1.0*unit.nanometers,
                                     constraints=app.HBonds)
    
    # Use a robust integrator for equilibration
    integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
    
    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    # --- 5. Energy Minimization ---
    print("Performing energy minimization...")
    simulation.minimizeEnergy()
    min_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    print(f"Minimized energy: {min_energy}")

    # --- 6. NPT Equilibration ---
    # Add a barostat for constant pressure simulation
    system.addForce(mm.MonteCarloBarostat(1*unit.bar, 300*unit.kelvin))
    # Re-initialize the simulation with the barostat
    simulation.context.reinitialize(preserveState=True)

    print("Running NPT equilibration...")
    simulation.step(50000) # Run for 100 ps

    # --- 7. Save the Equilibrated System ---
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    
    # Save as PDB
    with open(output_pdb, 'w') as f:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), f)
    
    # Save as XML for easy reloading in OpenMM
    with open(output_xml, 'w') as f:
        f.write(mm.XmlSerializer.serialize(state))

    print(f"Equilibrated system saved to {output_pdb} and {output_xml}")

if __name__ == "__main__":
    setup_md_system()

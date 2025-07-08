
This script runs a reactive molecular dynamics simulation to calculate
ab-initio kinetic rate constants for key reactions in polyethylene degradation.
"""
import numpy as np
import json
from pathlib import Path
import openmm as mm
from openmm import app, unit

# --- Constants for Eyring Equation ---
k_B = 1.380649e-23 * unit.joule/unit.kelvin  # Boltzmann constant
h = 6.62607015e-34 * unit.joule * unit.second # Planck constant
R = 8.314462618 * unit.joule/unit.mole/unit.kelvin # Ideal gas constant

def calculate_activation_energy_qmmm(system, reaction_type: str) -> unit.Quantity:
    """
    Placeholder function for a full QM/MM activation energy calculation.

    In a real implementation, this function would:
    1.  Identify the atoms involved in the specified reaction (e.g., two PEOO* radicals).
    2.  Define the QM region (the reacting atoms) and the MM region (the rest of the system).
    3.  Interface with a quantum chemistry engine (e.g., xTB, ORCA, Gaussian).
    4.  Perform a transition state search (e.g., Nudged Elastic Band or Berny optimization)
        to find the energy barrier for the reaction.
    5.  Return the calculated activation energy (enthalpy of activation, dH).

    Args:
        system: The OpenMM system object.
        reaction_type (str): The type of reaction ('crosslinking' or 'scission').

    Returns:
        unit.Quantity: A placeholder activation energy in kJ/mol.
    """
    print(f"--- Running Placeholder QM/MM for {reaction_type} ---")
    print("NOTE: This is NOT a real QM/MM calculation.")
    
    if reaction_type == "crosslinking":
        # Plausible activation energy for radical-radical combination, which is typically low.
        activation_energy = 25.0 * unit.kilojoules_per_mole
    elif reaction_type == "scission":
        # Plausible activation energy for C-C bond scission, which is typically high.
        activation_energy = 150.0 * unit.kilojoules_per_mole
    else:
        raise ValueError("Unknown reaction type")
        
    print(f"Placeholder activation energy: {activation_energy}")
    return activation_energy

def eyring_equation(delta_H: unit.Quantity, T: unit.Quantity) -> float:
    """
    Calculates a rate constant (k) from activation enthalpy (delta_H) using the
    Eyring equation from transition state theory.

    This calculation assumes the transmission coefficient is 1 and that the
    entropy of activation (delta_S) is negligible, which is a common simplification.

    k = (k_B * T / h) * exp(-delta_H / (R * T))

    Args:
        delta_H (unit.Quantity): Enthalpy of activation (e.g., in kJ/mol).
        T (unit.Quantity): Temperature (e.g., in Kelvin).

    Returns:
        float: The calculated first-order rate constant in 1/s.
    """
    exponent = -delta_H / (R * T)
    pre_factor = (k_B * T) / h
    rate_constant = pre_factor * np.exp(exponent)
    return rate_constant.value_in_unit(unit.second**-1)

def run_reactive_md():
    """
    Main function to orchestrate the reactive MD simulation.
    """
    print("--- Running Reactive MD for Rate Constant Calculation ---")
    
    # --- 1. Load the Equilibrated System ---
    system_xml_path = Path("data/equilibrated_pe_system.xml")
    if not system_xml_path.exists():
        print(f"Error: {system_xml_path} not found. Please run setup_md_system.py first.")
        return
        
    with open(system_xml_path, 'r') as f:
        system_state = mm.XmlSerializer.deserialize(f.read())

    # --- 2. Calculate Activation Energies (using placeholder) ---
    Ea_crosslinking = calculate_activation_energy_qmmm(system_state, "crosslinking")
    Ea_scission = calculate_activation_energy_qmmm(system_state, "scission")

    # --- 3. Convert to Rate Constants using Eyring Equation ---
    temp = 300 * unit.kelvin
    k_crosslink = eyring_equation(Ea_crosslinking, temp)
    k_scission = eyring_equation(Ea_scission, temp)

    print(f"\nCalculated rate constant for crosslinking at {temp}: {k_crosslink:.4e} 1/s")
    print(f"Calculated rate constant for scission at {temp}: {k_scission:.4e} 1/s")

    # --- 4. Save the Ab-Initio Parameters ---
    ab_initio_params = {
        "temperature_K": temp.value_in_unit(unit.kelvin),
        "activation_energy_crosslinking_kJ_mol": Ea_crosslinking.value_in_unit(unit.kilojoules_per_mole),
        "activation_energy_scission_kJ_mol": Ea_scission.value_in_unit(unit.kilojoules_per_mole),
        "k_crosslink_s-1": k_crosslink,
        "k_scission_s-1": k_scission
    }

    output_path = Path("models/ab_initio_params.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(ab_initio_params, f, indent=4)
        
    print(f"\nAb-initio rate constants saved to {output_path}")

if __name__ == "__main__":
    run_reactive_md()

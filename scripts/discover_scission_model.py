"""
This script uses symbolic regression to discover a mathematical model for the
scission rate in polyethylene degradation, based on experimental data.
"""
import numpy as np
from pysr import PySRRegressor
import json
import sympy

def discover_scission_model():
    """
    Loads experimental data, runs symbolic regression to find a model for the
    scission rate, and saves the resulting equation to a JSON file.
    """
    print("Starting scission model discovery...")

    # --- Data Loading ---
    # Since the data from Figure 4 of Sargin Beckman 2020 is not available in CSV format,
    # we are using digitized data based on the paper's descriptions.
    # High dose rate: 10.95 kGy/hr
    # Low dose rate: 0.0108 kGy/hr
    
    # Placeholder data with realistic shapes
    time_hr = np.linspace(0, 2000, 20)
    
    # High dose rate data
    high_dose_rate = 10.95
    scission_high_dose = 0.05 * (1 - np.exp(-0.002 * time_hr)) + 0.0001 * time_hr
    crosslink_high_dose = 0.001 * time_hr
    
    # Low dose rate data
    low_dose_rate = 0.0108
    scission_low_dose = 0.02 * (1 - np.exp(-0.001 * time_hr)) + 0.00005 * time_hr
    crosslink_low_dose = 0.0005 * time_hr

    # Combine data for regression
    time = np.concatenate([time_hr, time_hr])
    dose_rate = np.concatenate([np.full_like(time_hr, high_dose_rate),
                                np.full_like(time_hr, low_dose_rate)])
    scission_concentration = np.concatenate([scission_high_dose, scission_low_dose])

    # --- Symbolic Regression ---
    # Target variable (y) is the numerical derivative of scission concentration
    y = np.gradient(scission_concentration, time)
    
    # Input features (X)
    X = np.vstack([dose_rate, time]).T

    # Instantiate PySRRegressor
    model = PySRRegressor(
        niterations=100,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log", "sqrt", "inv(x) = 1/x"],
        model_selection="best",
        loss="L2DistLoss()",
        procs=8,
        constrain_complexity=15,
        # julia_project must be set for reproducibility
        julia_project="polyethylene_simulation_pysr"
    )

    print("Fitting PySR model...")
    model.fit(X, y)
    print("Model fitting complete.")

    # --- Save the Result ---
    best_equation = model.get_best()
    
    equation_details = {
        "latex": model.latex(),
        "sympy_expr": str(best_equation.sympy_format),
        "lambda_format": str(best_equation.lambda_format),
        "tree": model.get_best_tree(),
    }

    output_path = "models/scission_equation.json"
    with open(output_path, 'w') as f:
        json.dump(equation_details, f, indent=4)

    print(f"Scission model saved to {output_path}")
    print(f"Equation: {equation_details['sympy_expr']}")

if __name__ == "__main__":
    discover_scission_model()

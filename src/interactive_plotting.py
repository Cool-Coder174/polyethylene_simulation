"""
This module provides functions for generating interactive plots of simulation data
using Plotly. It fetches data from the SQLite database and visualizes chemical
species concentrations and reaction rates over time.
"""

import pandas as pd
import plotly.express as px
from pathlib import Path
from .database import fetch_data_for_plotting # Updated to relative import

def create_interactive_plots(db_path: Path, output_dir: Path, run_id: str):
    """
    Generates and saves interactive HTML plots for chemical kinetics data
    from a specific simulation run.

    This function creates two types of plots:
    1. Average Chemical Species Concentration vs. Time: Shows how the average
       concentrations of different chemical species change over the simulation's
       time steps.
    2. Relative Rates of Crosslinking vs. Scission: Illustrates the change
       in crosslinking and scission product concentrations over time, giving
       an indication of their relative reaction rates.

    The plots are saved as HTML files in the specified output directory,
    allowing for interactive exploration in a web browser.

    Args:
        db_path (Path): The absolute path to the SQLite database file.
        output_dir (Path): The directory where the generated HTML plots will be saved.
        run_id (str): The unique identifier of the simulation run for which to generate plots.
    """
    # Fetch simulation data for the given run_id from the database.
    df = fetch_data_for_plotting(db_path, run_id)

    # If no data is returned (e.g., invalid run_id or empty run), exit the function.
    if df.empty:
        print(f"No data found for run_id: {run_id}. Skipping plot generation.")
        return

    # Ensure the output directory exists. If not, create it.
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Chemical Species Concentration vs. Time ---
    # Define the list of chemical species columns to be plotted.
    species_to_plot = [
        'pe_conc',       # Polyethylene concentration
        'o2_conc',       # Oxygen concentration
        'pe_rad_conc',   # Alkyl Radical concentration
        'peoo_rad_conc', # Peroxy Radical concentration
        'peooh_conc',    # Hydroperoxide concentration
        'peoope_conc',   # Crosslink concentration
        'pecooh_conc'    # Scission concentration
    ]

    # Group the DataFrame by 'time_step' and calculate the mean concentration
    # for each species. This provides an average view across all voxels.
    df_avg = df.groupby('time_step')[species_to_plot].mean().reset_index()

    # Create a line plot using Plotly Express.
    # x-axis: time_step, y-axis: concentrations of selected species.
    fig_conc = px.line(df_avg, x='time_step', y=species_to_plot,
                       title=f'Average Chemical Species Concentration vs. Time (Run: {run_id[:8]})',
                       labels={'value': 'Concentration (mol/L)', 'variable': 'Species', 'time_step': 'Time (s)'},
                       template='plotly_white') # Use a clean white template for the plot

    # Save the concentration plot as an HTML file.
    # The filename includes a truncated run_id for easy identification.
    fig_conc.write_html(output_dir / f"concentrations_{run_id[:8]}.html")

    # --- Plot 2: Crosslinking vs. Scission Rates ---
    # Calculate the change in crosslink and scission concentrations between consecutive time steps.
    # This approximates the rate of formation for these products.
    df_avg['crosslink_rate'] = df_avg['peoope_conc'].diff().fillna(0)
    df_avg['scission_rate'] = df_avg['pecooh_conc'].diff().fillna(0)

    # Create a line plot for the calculated rates.
    fig_rates = px.line(df_avg, x='time_step', y=['crosslink_rate', 'scission_rate'],
                        title=f'Relative Rates of Crosslinking vs. Scission (Run: {run_id[:8]})',
                        labels={'value': 'Rate (Δmol/L/s)', 'variable': 'Reaction', 'time_step': 'Time (s)'},
                        template='plotly_white')

    # Save the rates plot as an HTML file.
    fig_rates.write_html(output_dir / f"rates_{run_id[:8]}.html")
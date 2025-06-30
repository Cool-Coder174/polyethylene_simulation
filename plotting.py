# plotting.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Use a professional plot style
plt.style.use('seaborn-v0_8-whitegrid')

def generate_and_save_plots(dataframe: pd.DataFrame, base_output_dir: Path):
    """
    Orchestrator function to generate and save all required correlation plots.
    
    Args:
        dataframe (pd.DataFrame): The simulation results data.
        base_output_dir (Path): The main output directory.
    """
    if dataframe.empty:
        print("Warning: Cannot generate plots from empty dataframe.")
        return
        
    # Create the specific subdirectory for these plots
    plot_dir = base_output_dir / "pCorrelations"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    run_id = dataframe['simulation_run_id'].iloc[0]

    # --- Call individual helper functions for each plot ---
    plot_connectivity_vs_events(dataframe, plot_dir, run_id)
    plot_eigenvalue_vs_events(dataframe, plot_dir, run_id)
    plot_connectivity_vs_eigenvalue(dataframe, plot_dir, run_id)
    plot_combined_metrics_vs_time(dataframe, plot_dir, run_id)
    # Add calls for the "single-process" plots if needed, though they require
    # dedicated simulation runs to isolate the effects. The combined plots are more general.

def plot_connectivity_vs_events(df, output_dir, run_id):
    """Plots Node Connectivity vs. Crosslinking and Scission Percentages."""
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=df, x='crosslinking_pct', y='avg_node_connectivity', label='Crosslinking Effect', color='blue', alpha=0.7)
    sns.scatterplot(data=df, x='scission_pct', y='avg_node_connectivity', label='Scission Effect', color='red', alpha=0.7)
    plt.title('Node Connectivity vs. Event Percentage')
    plt.xlabel('Event Percentage (%)')
    plt.ylabel('Average Node Connectivity (Degree)')
    plt.legend()
    plt.savefig(output_dir / f"conn_vs_events_{run_id[:8]}.png", dpi=150)
    plt.close()

def plot_eigenvalue_vs_events(df, output_dir, run_id):
    """Plots Fiedler Value vs. Crosslinking and Scission Percentages."""
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=df, x='crosslinking_pct', y='graph_laplacian_l2', label='Crosslinking Effect', color='blue', alpha=0.7)
    sns.scatterplot(data=df, x='scission_pct', y='graph_laplacian_l2', label='Scission Effect', color='red', alpha=0.7)
    plt.title('Fiedler Value (l2) vs. Event Percentage')
    plt.xlabel('Event Percentage (%)')
    plt.ylabel('Fiedler Value (l2)')
    plt.legend()
    plt.savefig(output_dir / f"l2_vs_events_{run_id[:8]}.png", dpi=150)
    plt.close()

def plot_connectivity_vs_eigenvalue(df, output_dir, run_id):
    """Plots a direct correlation between Node Connectivity and Fiedler Value."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='avg_node_connectivity', y='graph_laplacian_l2', alpha=0.6, s=50)
    plt.title('Node Connectivity vs. Fiedler Value (l2)')
    plt.xlabel('Average Node Connectivity (Degree)')
    plt.ylabel('Fiedler Value (l2)')
    plt.savefig(output_dir / f"conn_vs_l2_{run_id[:8]}.png", dpi=150)
    plt.close()

def plot_combined_metrics_vs_time(df, output_dir, run_id):
    """Creates a multi-panel plot of Connectivity and Fiedler Value vs. Time."""
    fig, axs = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Evolution of Connectivity Metrics Over Time', fontsize=16)
    
    # Plot 1: Node Connectivity vs. Time
    sns.lineplot(data=df, x='day', y='avg_node_connectivity', ax=axs[0], color='tab:blue', errorbar=('ci', 95))
    axs[0].set_title('Average Node Connectivity')
    axs[0].set_ylabel('Average Degree')
    
    # Plot 2: Fiedler Value vs. Time
    sns.lineplot(data=df, x='day', y='graph_laplacian_l2', ax=axs[1], color='tab:red', errorbar=('ci', 95))
    axs[1].set_title('Fiedler Value (Algebraic Connectivity)')
    axs[1].set_xlabel('Simulation Day')
    axs[1].set_ylabel('Eigenvalue (l2)')
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_dir / f"combined_metrics_vs_time_{run_id[:8]}.png", dpi=150)
    plt.close()

# Note: Plots for "scission only" or "crosslink only" would require running the
# simulation with one of the event types disabled to isolate its effect. The
# current structure is set up for combined effects, which is more general.


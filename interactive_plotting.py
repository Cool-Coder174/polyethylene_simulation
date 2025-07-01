# interactive_plotting.py
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from database import fetch_data_for_plotting

def create_interactive_plots(db_path: Path, output_dir: Path, run_id: str):
    """
    Generates and saves a suite of interactive plots for a given simulation run.
    """
    df = fetch_data_for_plotting(db_path, run_id)
    if df.empty:
        return

    output_dir.mkdir(exist_ok=True)

    # --- Time Series Plot ---
    fig_ts = px.line(df, x='day', y=['avg_node_connectivity', 'graph_laplacian_l2'],
                     title=f'Metrics vs. Time (Run: {run_id[:8]})',
                     labels={'value': 'Metric Value', 'variable': 'Metric'},
                     template='plotly_white')
    fig_ts.write_html(output_dir / f"timeseries_{run_id[:8]}.html")

    # --- Correlation Scatter Plot ---
    fig_corr = px.scatter(df, x='avg_node_connectivity', y='graph_laplacian_l2',
                          color='day',
                          title=f'Connectivity vs. Fiedler Value (Run: {run_id[:8]})',
                          template='plotly_white')
    fig_corr.write_html(output_dir / f"correlation_{run_id[:8]}.html")


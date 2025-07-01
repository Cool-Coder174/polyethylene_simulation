# database.py
import sqlite3
import pandas as pd
from pathlib import Path

def init_database(db_path: Path):
    """
    Initializes the SQLite database and creates the necessary tables if they don't exist.
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # Main table to store time-series data from each simulation run
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS simulation_data (
                run_id TEXT,
                optuna_trial_id INTEGER,
                episode_num INTEGER,
                day INTEGER,
                chain_length_avg REAL,
                num_chains INTEGER,
                avg_node_connectivity REAL,
                crosslinking_pct REAL,
                scission_pct REAL,
                graph_laplacian_l2 REAL,
                PRIMARY KEY (run_id, day)
            )
        """)
        # Table to store run-level metadata and final results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS run_metadata (
                run_id TEXT PRIMARY KEY,
                optuna_trial_id INTEGER,
                episode_num INTEGER,
                start_time TEXT,
                end_time TEXT,
                duration_seconds REAL,
                initial_params TEXT,
                final_reward REAL
            )
        """)
        conn.commit()

def log_simulation_data(db_path: Path, data_df: pd.DataFrame):
    """
    Logs a DataFrame of time-series simulation data to the database.
    """
    with sqlite3.connect(db_path) as conn:
        data_df.to_sql('simulation_data', conn, if_exists='append', index=False)

def log_run_metadata(db_path: Path, metadata: dict):
    """
    Logs the metadata for a completed simulation run.
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO run_metadata (run_id, optuna_trial_id, episode_num, start_time, end_time, duration_seconds, initial_params, final_reward)
            VALUES (:run_id, :optuna_trial_id, :episode_num, :start_time, :end_time, :duration_seconds, :initial_params, :final_reward)
        """, metadata)
        conn.commit()

def fetch_data_for_plotting(db_path: Path, run_id: str = None):
    """
    Fetches simulation data for a specific run or all runs.
    """
    with sqlite3.connect(db_path) as conn:
        query = "SELECT * FROM simulation_data"
        if run_id:
            query += f" WHERE run_id = '{run_id}'"
        df = pd.read_sql_query(query, conn)
        return df


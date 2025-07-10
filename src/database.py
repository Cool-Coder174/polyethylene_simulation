"""
This module handles all database interactions for the polyethylene degradation simulation.
It uses SQLite to store simulation time-series data and run-level metadata.
"""

import sqlite3
import pandas as pd
from pathlib import Path

def init_database(db_path: Path):
    """
    Initializes the SQLite database at the specified path.

    This function creates two tables:
    1. `simulation_data`: Stores detailed time-series data for each voxel,
       including concentrations of various chemical species.
    2. `run_metadata`: Stores metadata for each simulation run, such as
       start/end times, duration, initial dose rate, final reward, and status.

    Args:
        db_path (Path): The absolute path to the SQLite database file.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Create the main table to store time-series data for each voxel.
            # This table records the concentration of different chemical species
            # at specific time steps within individual voxels.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS simulation_data (
                    run_id TEXT,            -- Unique identifier for each simulation run
                    time_step INTEGER,      -- The discrete time step in the simulation
                    voxel_id INTEGER,       -- Unique identifier for each spatial voxel
                    pe_conc REAL,           -- Polyethylene concentration
                    o2_conc REAL,           -- Oxygen concentration
                    pe_rad_conc REAL,       -- Alkyl Radical concentration
                    peoo_rad_conc REAL,     -- Peroxy Radical concentration
                    peooh_conc REAL,        -- Hydroperoxide concentration
                    peoope_conc REAL,       -- Crosslink concentration (representing crosslinking events)
                    pecooh_conc REAL,       -- Scission concentration (representing chain scission events)
                    PRIMARY KEY (run_id, time_step, voxel_id) -- Composite primary key for unique records
                )
            """)

            # Create a table to store run-level metadata.
            # This table provides an overview and summary statistics for each
            # complete simulation run.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS run_metadata (
                    run_id TEXT PRIMARY KEY,    -- Unique identifier for the simulation run (matches simulation_data.run_id)
                    optuna_trial_id INTEGER,    -- ID from Optuna if hyperparameter optimization is used
                    start_time TEXT,            -- Timestamp when the run started
                    end_time TEXT,              -- Timestamp when the run ended
                    duration_seconds REAL,      -- Total duration of the run in seconds
                    initial_dose_rate REAL,     -- The initial radiation dose rate applied
                    final_reward REAL,          -- The final reward obtained by the RL agent for this run
                    status TEXT                 -- Status of the run (e.g., 'completed', 'failed', 'pruned')
                )
            """)
            conn.commit() # Commit the changes to create the tables
    except sqlite3.Error as e:
        logging.error(f"Database error in init_database: {e}")
        raise

def log_simulation_data(db_path: Path, data_df: pd.DataFrame):
    """
    Logs a DataFrame containing voxel-level chemical concentration data into the
    `simulation_data` table.

    Args:
        db_path (Path): The absolute path to the SQLite database file.
        data_df (pd.DataFrame): A Pandas DataFrame where each row represents
                                a record for `simulation_data` table.
                                It must have columns matching the table schema.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            # Use pandas to efficiently append the DataFrame to the 'simulation_data' table.
            # if_exists='append' ensures new data is added without overwriting existing data.
            data_df.to_sql('simulation_data', conn, if_exists='append', index=False)
    except sqlite3.Error as e:
        logging.error(f"Database error in log_simulation_data: {e}")
        raise

def log_run_metadata(db_path: Path, metadata: dict):
    """
    Logs the metadata for a completed or terminated simulation run into the
    `run_metadata` table.

    Args:
        db_path (Path): The absolute path to the SQLite database file.
        metadata (dict): A dictionary containing the metadata for a single run.
                         Keys must match the column names in the `run_metadata` table.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Insert the metadata using a parameterized query to prevent SQL injection.
            cursor.execute("""
                INSERT INTO run_metadata (run_id, optuna_trial_id, start_time, end_time, duration_seconds, initial_dose_rate, final_reward, status)
                VALUES (:run_id, :optuna_trial_id, :start_time, :end_time, :duration_seconds, :initial_dose_rate, :final_reward, :status)
            """, metadata)
            conn.commit() # Commit the changes to save the metadata
    except sqlite3.Error as e:
        logging.error(f"Database error in log_run_metadata: {e}")
        raise

def fetch_data_for_plotting(db_path: Path, run_id: str = None) -> pd.DataFrame:
    """
    Fetches simulation data from the `simulation_data` table for plotting.
    Can fetch data for a specific run or for all runs if no `run_id` is provided.

    Args:
        db_path (Path): The absolute path to the SQLite database file.
        run_id (str, optional): The unique identifier of the simulation run
                                to fetch data for. If None, data for all runs
                                will be fetched. Defaults to None.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the fetched simulation data.
                      Returns an empty DataFrame if no data is found.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            query = "SELECT * FROM simulation_data"
            params = None
            if run_id:
                query += " WHERE run_id = ?"
                params = (run_id,)
            df = pd.read_sql_query(query, conn, params=params)
            return df
    except sqlite3.Error as e:
        print(f"Database error in fetch_data_for_plotting: {e}")
        raise

def get_all_run_ids(db_path: Path) -> list[str]:
    """
    Retrieves a list of all unique run_ids from the `run_metadata` table.

    Args:
        db_path (Path): The absolute path to the SQLite database file.

    Returns:
        list[str]: A list of all unique run_id strings.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT run_id FROM run_metadata")
            # Fetch all results and flatten the list of tuples into a simple list of strings.
            run_ids = [item[0] for item in cursor.fetchall()]
            return run_ids
    except sqlite3.Error as e:
        print(f"Database error in get_all_run_ids: {e}")
        raise

def get_run_metadata(db_path: Path, run_id: str) -> dict:
    """
    Retrieves the metadata for a specific run_id from the `run_metadata` table.

    Args:
        db_path (Path): The absolute path to the SQLite database file.
        run_id (str): The unique identifier of the run to retrieve metadata for.

    Returns:
        dict: A dictionary containing the metadata for the specified run.
              Returns an empty dictionary if the run_id is not found.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row # Set row_factory to get dict-like rows
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM run_metadata WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if row:
                return dict(row) # Convert the row object to a dictionary
            return {} # Return an empty dict if no record was found
    except sqlite3.Error as e:
        print(f"Database error in get_run_metadata: {e}")
        raise

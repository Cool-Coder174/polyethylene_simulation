import sqlite3
import unittest
from pathlib import Path
import pandas as pd
from src.database import (
    init_database,
    log_simulation_data,
    log_run_metadata,
    fetch_data_for_plotting,
    get_all_run_ids,
    get_run_metadata,
)

class TestDatabase(unittest.TestCase):
    def setUp(self):
        """Set up a temporary database for testing."""
        self.db_path = Path("test_database.db")
        init_database(self.db_path)

    def tearDown(self):
        """Remove the temporary database after testing."""
        self.db_path.unlink(missing_ok=True)

    def test_init_database(self):
        """Test that the database and tables are created."""
        self.assertTrue(self.db_path.exists())
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='simulation_data'")
            self.assertIsNotNone(cursor.fetchone())
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='run_metadata'")
            self.assertIsNotNone(cursor.fetchone())

    def test_log_and_fetch_simulation_data(self):
        """Test logging and fetching simulation data."""
        data = {
            'run_id': ['test_run'] * 2,
            'time_step': [1, 2],
            'voxel_id': [101, 102],
            'pe_conc': [0.1, 0.2],
            'o2_conc': [0.3, 0.4],
            'pe_rad_conc': [0.01, 0.02],
            'peoo_rad_conc': [0.03, 0.04],
            'peooh_conc': [0.05, 0.06],
            'peoope_conc': [0.07, 0.08],
            'pecooh_conc': [0.09, 0.10],
        }
        df = pd.DataFrame(data)
        log_simulation_data(self.db_path, df)

        fetched_df = fetch_data_for_plotting(self.db_path, run_id='test_run')
        self.assertEqual(len(fetched_df), 2)
        self.assertEqual(list(fetched_df.columns), list(df.columns))

    def test_log_and_get_run_metadata(self):
        """Test logging and fetching run metadata."""
        metadata = {
            'run_id': 'test_run_2',
            'optuna_trial_id': 1,
            'start_time': '2025-07-10 10:00:00',
            'end_time': '2025-07-10 11:00:00',
            'duration_seconds': 3600,
            'initial_dose_rate': 0.5,
            'final_reward': 0.99,
            'status': 'completed',
        }
        log_run_metadata(self.db_path, metadata)

        fetched_metadata = get_run_metadata(self.db_path, 'test_run_2')
        self.assertEqual(fetched_metadata['run_id'], 'test_run_2')
        self.assertEqual(fetched_metadata['final_reward'], 0.99)

    def test_get_all_run_ids(self):
        """Test fetching all run IDs."""
        metadata1 = {
            'run_id': 'run1',
            'optuna_trial_id': 1, 'start_time': 'a', 'end_time': 'b', 'duration_seconds': 1, 'initial_dose_rate': 1, 'final_reward': 1, 'status': 'c'
        }
        metadata2 = {
            'run_id': 'run2',
            'optuna_trial_id': 2, 'start_time': 'd', 'end_time': 'e', 'duration_seconds': 2, 'initial_dose_rate': 2, 'final_reward': 2, 'status': 'f'
        }
        log_run_metadata(self.db_path, metadata1)
        log_run_metadata(self.db_path, metadata2)

        run_ids = get_all_run_ids(self.db_path)
        self.assertIn('run1', run_ids)
        self.assertIn('run2', run_ids)
        self.assertEqual(len(run_ids), 2)

if __name__ == '__main__':
    unittest.main()
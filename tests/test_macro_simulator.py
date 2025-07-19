import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import pandas as pd
import settings
from utils.macro_simulator import ECBPoolSimulator

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Import your class here
# from macro_simulator import ECBPoolSimulator

class TestECBPoolSimulator(unittest.TestCase):
    @patch('utils.macro_simulator.Fred')
    @patch('utils.macro_simulator.yf')
    def test_simulator_workflow(self, mock_yf, mock_fred):
        # Mock FRED series fetching
        fred_instance = mock_fred.return_value
        dummy_index = pd.date_range("2000-01-01", periods=50, freq="Q")
        dummy_series = pd.Series(np.random.rand(50), index=dummy_index)
        fred_instance.get_series.return_value = dummy_series

        # Mock Yahoo Finance
        spx_mock = MagicMock()
        history_df = pd.DataFrame({'Close': np.random.rand(50)}, index=dummy_index)
        spx_mock.history.return_value = history_df
        mock_yf.Ticker.return_value = spx_mock

        # Initialize simulator with mocks
        sim = ECBPoolSimulator(fred_api_key='dummy', debug=True)

        # Reset and run simulation
        sim.reset(path_length=5, n_draws=2, pool_weights={'VAR':0.5, 'DSGE':0.5})
        self.assertIsNotNone(sim.simulated_path)
        self.assertEqual(len(sim.simulated_path), 6) # path_length+1 (includes origin)

        origin_data = sim.retrieve_origin_data()
        self.assertIsInstance(origin_data, pd.DataFrame)
        self.assertEqual(origin_data.shape[0], 1)

        forecast = sim.retrieve_next_forecast()
        self.assertTrue(all([col in forecast.index for col in sim.default_output_columns]))

if __name__ == "__main__":
    unittest.main()

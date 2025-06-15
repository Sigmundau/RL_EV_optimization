import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Adjust the path to import the Simulation class
# Assuming macro_simulator.py is in the same directory as the test file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from utils.macro_simulator import Simulation

class TestSimulation(unittest.TestCase):

    def setUp(self):
        # Set a dummy FRED API key for testing
        self.fred_api_key = "5446f1dec140788657af4c0720a225b2"
        self.var_order = 1 # Simple VAR order for testing

        # Mock data for FRED and Yahoo Finance
        # Create longer time series to ensure adequate data after quarterly resampling
        date_range = pd.date_range(start='2000-01-01', periods=200, freq='MS')
        quarterly_range = pd.date_range(start='2000-01-01', periods=50, freq='QS')
        
        self.mock_fred_data = {
            'AAAFF': pd.Series(np.random.rand(200) * 0.5, index=date_range),
            'AAA10YM': pd.Series(np.random.rand(200) * 1.0, index=date_range),
            'BAAFF': pd.Series(np.random.rand(200) * 0.7, index=date_range),
            'BAA10YM': pd.Series(np.random.rand(200) * 1.2, index=date_range),
            'CE16OV': pd.Series(np.random.rand(200) * 100 + 150000, index=date_range),
            'FEDFUNDS': pd.Series(np.random.rand(200) * 0.2 + 0.05, index=date_range),
            'FPI': pd.Series(np.random.rand(200) * 500 + 1000, index=date_range),
            'GDP': pd.Series(np.random.rand(50) * 5000 + 15000, index=quarterly_range),
            'GDPDEF': pd.Series(np.random.rand(50) * 5 + 100, index=quarterly_range),
            'GS10': pd.Series(np.random.rand(200) * 2 + 1, index=date_range),
            'M2SL': pd.Series(np.random.rand(200) * 1000 + 10000, index=date_range),
            'PCEC': pd.Series(np.random.rand(200) * 1000 + 5000, index=date_range),
            'UNRATE': pd.Series(np.random.rand(200) * 3 + 3, index=date_range),
            'PCEPI': pd.Series(np.random.rand(200) * 10 + 90, index=date_range),
            'GS2': pd.Series(np.random.rand(200) * 1 + 0.5, index=date_range),
            'TWEXB': pd.Series(np.random.rand(200) * 20 + 80, index=date_range)
        }

    def test_initialization_and_data_preparation(self):
        """
        Test if the Simulation class initializes correctly,
        and if _fetch_and_prepare_data processes data as expected.
        """
        with patch('macro_simulator.Fred') as mock_fred_class, \
             patch('macro_simulator.VAR') as mock_var_class:
            
            # Setup FRED mock
            mock_fred_instance = MagicMock()
            mock_fred_class.return_value = mock_fred_instance
            
            def get_series_side_effect(series_id):
                series_map = {
                    'AAAFF': 'AAAFF', 'AAA10YM': 'AAA10YM', 'BAAFF': 'BAAFF', 'BAA10YM': 'BAA10YM',
                    'CE16OV': 'CE16OV', 'FEDFUNDS': 'FEDFUNDS', 'FPI': 'FPI', 'GDP': 'GDP',
                    'GDPDEF': 'GDPDEF', 'GS10': 'GS10', 'M2SL': 'M2SL', 'PCEC': 'PCEC',
                    'UNRATE': 'UNRATE', 'PCEPI': 'PCEPI', 'GS2': 'GS2', 'TWEXB': 'TWEXB'
                }
                if series_id in series_map:
                    return self.mock_fred_data[series_map[series_id]]
                else:
                    raise ValueError(f"Series ID {series_id} not found in mock data.")
            
            mock_fred_instance.get_series.side_effect = get_series_side_effect
            
            # Setup VAR mock
            mock_var_instance = MagicMock()
            mock_var_class.return_value = mock_var_instance
            mock_var_fit_instance = MagicMock()
            mock_var_instance.fit.return_value = mock_var_fit_instance
            
            num_vars = 16
            mock_var_fit_instance.params = np.random.rand(num_vars * self.var_order + 1, num_vars)
            mock_var_fit_instance.sigma_u = np.eye(num_vars) * 0.01
            mock_var_fit_instance.forecast.return_value = np.random.rand(1, num_vars)
            
            # Mock the random functions used in reset
            with patch('macro_simulator.np.random.randint') as mock_randint, \
                 patch('macro_simulator.np.random.multivariate_normal') as mock_multivariate_normal:
                
                mock_randint.return_value = 10
                mock_multivariate_normal.return_value = np.random.rand(100, num_vars) * 0.01
                
                # Initialize Simulation
                simulation = Simulation(fred_api_key=self.fred_api_key, var_order=self.var_order, debug=True)
                
                # Assertions
                mock_fred_class.assert_called_once_with(api_key=self.fred_api_key)
                self.assertIsNotNone(simulation.processed_data)
                self.assertFalse(simulation.processed_data.empty)
                
                # Check for expected columns
                expected_fred_cols = ['AAAFF', 'AAA10YM', 'BAAFF', 'BAA10YM', 'EMPLOYMENT', 'EFFR', 'FPI', 'GDP', 'GDP_DEF','10Y', 'M2', 'PCE', 'UNEMPLOYMENT', 'PCE_PRICE_INDEX', '2Y', 'USD_INDEX']
                for col in expected_fred_cols:
                    self.assertIn(col, simulation.processed_data.columns)
                
                self.assertIsNotNone(simulation.model_fit)

    def test_reset_path_generation(self):
        """
        Test the reset method for path generation and derived variable calculation.
        """
        with patch('macro_simulator.Fred') as mock_fred_class, \
             patch('macro_simulator.VAR') as mock_var_class:
            
            # Setup FRED mock
            mock_fred_instance = MagicMock()
            mock_fred_class.return_value = mock_fred_instance
            
            def get_series_side_effect(series_id):
                series_map = {
                    'AAAFF': 'AAAFF', 'AAA10YM': 'AAA10YM', 'BAAFF': 'BAAFF', 'BAA10YM': 'BAA10YM',
                    'CE16OV': 'CE16OV', 'FEDFUNDS': 'FEDFUNDS', 'FPI': 'FPI', 'GDP': 'GDP',
                    'GDPDEF': 'GDPDEF', 'GS10': 'GS10', 'M2SL': 'M2SL', 'PCEC': 'PCEC',
                    'UNRATE': 'UNRATE', 'PCEPI': 'PCEPI', 'GS2': 'GS2', 'TWEXB': 'TWEXB'
                }
                if series_id in series_map:
                    return self.mock_fred_data[series_map[series_id]]
                else:
                    raise ValueError(f"Series ID {series_id} not found in mock data.")
            
            mock_fred_instance.get_series.side_effect = get_series_side_effect
            
            # Setup VAR mock
            mock_var_instance = MagicMock()
            mock_var_class.return_value = mock_var_instance
            mock_var_fit_instance = MagicMock()
            mock_var_instance.fit.return_value = mock_var_fit_instance
            
            num_vars = 16
            mock_var_fit_instance.params = np.random.rand(num_vars * self.var_order + 1, num_vars)
            mock_var_fit_instance.sigma_u = np.eye(num_vars) * 0.01
            
            def mock_forecast(y, steps):
                return np.random.rand(steps, num_vars)
            mock_var_fit_instance.forecast.side_effect = mock_forecast
            
            # Mock random functions with proper return values
            with patch('macro_simulator.np.random.randint') as mock_randint, \
                 patch('macro_simulator.np.random.multivariate_normal') as mock_multivariate_normal:
                
                mock_randint.return_value = 10
                
                # For initialization
                mock_multivariate_normal.return_value = np.random.rand(100, num_vars) * 0.01
                
                # Initialize simulation
                simulation = Simulation(fred_api_key=self.fred_api_key, var_order=self.var_order, debug=True)
                
                # Test reset with specific parameters
                path_length = 10
                mock_multivariate_normal.return_value = np.random.rand(path_length, num_vars) * 0.01
                
                simulation.reset(path_length=path_length, noise_scale=1.0)
                
                # Assertions
                self.assertIsNotNone(simulation.simulated_path)
                self.assertGreater(len(simulation.simulated_path), 0)
                self.assertFalse(simulation.simulated_path.empty)
                
                # Check that derived variables are calculated
                expected_derived_cols = ['REAL_GROWTH', 'FX_STRENGTH', 'INFLATION', 'AAA_2Y', 'AAA_10Y', 'BAA_2Y', 'BAA_10Y']
                for col in expected_derived_cols:
                    self.assertIn(col, simulation.simulated_path.columns)

    def test_retrieve_next_forecast(self):
        """
        Test the retrieve_next_forecast method for correct slicing and auto-reset.
        """
        with patch('macro_simulator.Fred') as mock_fred_class, \
             patch('macro_simulator.VAR') as mock_var_class, \
             patch('macro_simulator.np.random.randint') as mock_randint, \
             patch('macro_simulator.np.random.multivariate_normal') as mock_multivariate_normal:
            
            # Setup all mocks
            mock_fred_instance = MagicMock()
            mock_fred_class.return_value = mock_fred_instance
            
            def get_series_side_effect(series_id):
                series_map = {
                    'AAAFF': 'AAAFF', 'AAA10YM': 'AAA10YM', 'BAAFF': 'BAAFF', 'BAA10YM': 'BAA10YM',
                    'CE16OV': 'CE16OV', 'FEDFUNDS': 'FEDFUNDS', 'FPI': 'FPI', 'GDP': 'GDP',
                    'GDPDEF': 'GDPDEF', 'GS10': 'GS10', 'M2SL': 'M2SL', 'PCEC': 'PCEC',
                    'UNRATE': 'UNRATE', 'PCEPI': 'PCEPI', 'GS2': 'GS2', 'TWEXB': 'TWEXB'
                }
                if series_id in series_map:
                    return self.mock_fred_data[series_map[series_id]]
                else:
                    raise ValueError(f"Series ID {series_id} not found in mock data.")
            
            mock_fred_instance.get_series.side_effect = get_series_side_effect
            
            mock_var_instance = MagicMock()
            mock_var_class.return_value = mock_var_instance
            mock_var_fit_instance = MagicMock()
            mock_var_instance.fit.return_value = mock_var_fit_instance
            
            num_vars = 16
            mock_var_fit_instance.params = np.random.rand(num_vars * self.var_order + 1, num_vars)
            mock_var_fit_instance.sigma_u = np.eye(num_vars) * 0.01
            
            def mock_forecast(y, steps):
                return np.random.rand(steps, num_vars)
            mock_var_fit_instance.forecast.side_effect = mock_forecast
            
            mock_randint.return_value = 10
            mock_multivariate_normal.return_value = np.random.rand(100, num_vars) * 0.01
            
            # Initialize simulation
            simulation = Simulation(fred_api_key=self.fred_api_key, var_order=self.var_order)
            
            # Test normal retrieval
            steps = 2
            forecast = simulation.retrieve_next_forecast(steps=steps)
            
            # Check that we get the expected columns
            for col in simulation.default_output_columns:
                self.assertIn(col, forecast.columns)
            
            # Check that current_index is updated
            self.assertEqual(simulation.current_index, steps)

    def test_auto_reset_functionality(self):
        """
        Test that auto-reset works when path is exhausted.
        """
        with patch('macro_simulator.Fred') as mock_fred_class, \
             patch('macro_simulator.VAR') as mock_var_class, \
             patch('macro_simulator.np.random.randint') as mock_randint, \
             patch('macro_simulator.np.random.multivariate_normal') as mock_multivariate_normal:
            
            # Setup all mocks
            mock_fred_instance = MagicMock()
            mock_fred_class.return_value = mock_fred_instance
            
            def get_series_side_effect(series_id):
                series_map = {
                    'AAAFF': 'AAAFF', 'AAA10YM': 'AAA10YM', 'BAAFF': 'BAAFF', 'BAA10YM': 'BAA10YM',
                    'CE16OV': 'CE16OV', 'FEDFUNDS': 'FEDFUNDS', 'FPI': 'FPI', 'GDP': 'GDP',
                    'GDPDEF': 'GDPDEF', 'GS10': 'GS10', 'M2SL': 'M2SL', 'PCEC': 'PCEC',
                    'UNRATE': 'UNRATE', 'PCEPI': 'PCEPI', 'GS2': 'GS2', 'TWEXB': 'TWEXB'
                }
                if series_id in series_map:
                    return self.mock_fred_data[series_map[series_id]]
                else:
                    raise ValueError(f"Series ID {series_id} not found in mock data.")
            
            mock_fred_instance.get_series.side_effect = get_series_side_effect
            
            mock_var_instance = MagicMock()
            mock_var_class.return_value = mock_var_instance
            mock_var_fit_instance = MagicMock()
            mock_var_instance.fit.return_value = mock_var_fit_instance
            
            num_vars = 16
            mock_var_fit_instance.params = np.random.rand(num_vars * self.var_order + 1, num_vars)
            mock_var_fit_instance.sigma_u = np.eye(num_vars) * 0.01
            
            def mock_forecast(y, steps):
                return np.random.rand(steps, num_vars)
            mock_var_fit_instance.forecast.side_effect = mock_forecast
            
            mock_randint.return_value = 10
            mock_multivariate_normal.return_value = np.random.rand(100, num_vars) * 0.01
            
            # Initialize simulation
            simulation = Simulation(fred_api_key=self.fred_api_key, var_order=self.var_order)
            
            # Manually set the index to the end of the path to test auto-reset
            original_length = len(simulation.simulated_path)
            simulation.current_index = original_length
            
            # The next call should trigger a reset
            forecast = simulation.retrieve_next_forecast(steps=1)
            
            # Check that we got data (which means reset worked)
            self.assertIsNotNone(forecast)
            self.assertFalse(forecast.empty)
            self.assertEqual(simulation.current_index, 1)  # Should be 1 after retrieving 1 step


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
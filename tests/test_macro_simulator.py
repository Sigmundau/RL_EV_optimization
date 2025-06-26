import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import pandas as pd
import settings
from utils.macro_simulator import Simulation, Dynamic_Simulation

DATA_PATH = './data/FRED_data.csv'  

class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.sim = Simulation(fred_api_key=settings.fred_api_key, debug=False, fred_csv_path=DATA_PATH)

    def test_simulation_init_and_reset(self):
        self.assertIsNotNone(self.sim.processed_data)
        self.assertIsNotNone(self.sim.model_fit)
        self.assertIsInstance(self.sim.simulated_path, pd.DataFrame)
        self.sim.reset(path_length=10)
        self.assertEqual(len(self.sim.simulated_path), 10)

    def test_simulation_retrieve_origin_data(self):
        origin = self.sim.retrieve_origin_data()
        self.assertIsInstance(origin, pd.DataFrame)
        self.assertEqual(len(origin), 1)

    def test_simulation_retrieve_next_forecast(self):
        forecast = self.sim.retrieve_next_forecast(steps=3)
        self.assertIsInstance(forecast, pd.DataFrame)
        self.assertLessEqual(len(forecast), 3)
        self.assertTrue(all(col in forecast.columns for col in self.sim.default_output_columns))


class TestDynamicSimulation(unittest.TestCase):
    def setUp(self):
        self.sim = Dynamic_Simulation(fred_api_key=settings.fred_api_key, debug=False, fred_csv_path=DATA_PATH)

    def test_dynamic_simulation_init_and_reset(self):
        self.assertIsNotNone(self.sim.processed_data)
        self.assertIsNotNone(self.sim.model_fit)
        self.assertIsInstance(self.sim.simulated_path, pd.DataFrame)
        self.sim.reset(path_length=10)
        self.assertEqual(len(self.sim.simulated_path), 10)

    def test_dynamic_simulation_retrieve_origin_data(self):
        origin = self.sim.retrieve_origin_data()
        self.assertIsInstance(origin, pd.DataFrame)
        self.assertEqual(len(origin), 1)

    def test_dynamic_simulation_retrieve_next_forecast(self):
        forecast = self.sim.retrieve_next_forecast(steps=3)
        self.assertIsInstance(forecast, pd.DataFrame)
        self.assertLessEqual(len(forecast), 3)
        self.assertTrue(all(col in forecast.columns for col in self.sim.default_output_columns))

    def test_dynamic_simulation_get_model_summary(self):
        summary = self.sim.get_model_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn('var_order', summary)
        self.assertIn('data_shape', summary)
        self.assertIn('garch_specs', summary)

if __name__ == '__main__':
    unittest.main()
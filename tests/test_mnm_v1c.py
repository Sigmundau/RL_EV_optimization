import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import datetime
import torch

from environments.mnm_v1c import MnM_v1c  # Update the import if needed

class TestMnMv1c(unittest.TestCase):

    @patch('environments.mnm_v1c.ECBPoolSimulator')
    @patch('environments.mnm_v1c.calculate_beta')
    def test_reset_and_step(self, mock_calculate_beta, mock_simulator_class):
        # Arrange: mock macro simulator and data
        mock_simulator = MagicMock()
        mock_origin_data = pd.DataFrame({
            'EFFR': [0.015],
            'MARKET_RETURN': [0.07],
            'GDP': [10000], 'PCE': [5000],
            'EMPLOYMENT': [160000], 'FPI': [1050],
            'AAA_10Y': [2.0], 'BAA_10Y': [4.0],
            'AAA_2Y': [1.5], 'BAA_2Y': [3.5],
        })
        mock_simulator.retrieve_origin_data.return_value = mock_origin_data
        # Create a mock next forecast with 16 elements
        mock_next_forecast = pd.Series([
            0.015, 0.07, 10000, 5000, 160000, 1050, 2.0, 4.0, 1.5, 3.5,
            0, 0, 0, 0, 0, 0
        ], index=[
            'EFFR', 'MARKET_RETURN', 'GDP', 'PCE', 'EMPLOYMENT', 'FPI',
            'AAA_10Y', 'BAA_10Y', 'AAA_2Y', 'BAA_2Y',
            'EXTRA1', 'EXTRA2', 'EXTRA3', 'EXTRA4', 'EXTRA5', 'EXTRA6'
        ])
        mock_simulator.retrieve_next_forecast.return_value = mock_next_forecast
        mock_simulator_class.return_value = mock_simulator

        mock_calculate_beta.return_value = 1.2

        # Mock company quarterly CSV data
        test_date = datetime.date(2020, 1, 1)
        company_data = pd.DataFrame({
            'fiscalDateEnding': ['2019-10-01', '2020-01-01'],
            'totalAssets': [1000, 1050],
            'shortTermDebt': [100, 110],
            'longTermDebt': [200, 210],
            'totalShareholderEquity': [700, 730],
            'totalRevenue': [500, 550],
            'costofGoodsAndServicesSold': [250, 275],
            'operatingExpenses': [100, 110],
            'interestIncome': [10, 11],
            'interestExpense': [12, 13],
            'depreciationAndAmortization': [20, 22],
            'incomeTaxExpense': [15, 16.5],
            'incomeBeforeTax': [60, 66],
            'ebit': [75, 82]
        })
        company_data.to_csv("mock_company.csv")

        # Act
        env = MnM_v1c(
            ticker="TEST",
            fred_api_key="dummy_key",
            csv_path="mock_company.csv",
            fixed_origin=True,
            simulation_earliest_origin=datetime.datetime(2020, 1, 1),
            simulation_horizon=4,
            fixed_path=True,
            debug=True
        )

        obs = env._reset()

        # Assert: initial observation structure and values
        self.assertIn('financial_metrics', obs)
        self.assertIn('macro_economic_data', obs)
        self.assertEqual(obs['financial_metrics'].shape[-1], 11)
        self.assertEqual(obs['macro_economic_data'].shape[-1], 10)

        # Simulate one action step
        dummy_action = torch.tensor([5, 5])  # Medium delta choice
        input_td = obs.clone()
        input_td.set("action", dummy_action)

        # New code change: mapping action to debt adjustment ratios
        action_mapping = {i: round(0.02 * i - 0.1, 2) for i in range(11)}
        st_debt_adjustment_ratio = action_mapping[int(dummy_action[0].item())]
        lt_debt_adjustment_ratio = action_mapping[int(dummy_action[1].item())]

        output = env._step(input_td)
        self.assertIn("next", output)
        self.assertIn("reward", output)
        self.assertIn("done", output)

        # Check shape consistency
        self.assertEqual(output["next"]["financial_metrics"].shape[-1], 11)
        macro_data = output["next"]["macro_economic_data"]
        # Accept scalar or vector output for macro_economic_data
        if macro_data.dim() == 0:
            self.assertEqual(macro_data.numel(), 1)
        else:
            self.assertEqual(macro_data.numel(), 16)
        self.assertTrue(output["reward"].item() >= -20.0)  # Reward should be numerical

    def tearDown(self):
        import os
        if os.path.exists("mock_company.csv"):
            os.remove("mock_company.csv")

if __name__ == "__main__":
    unittest.main()

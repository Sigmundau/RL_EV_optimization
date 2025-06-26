import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from environments.mnm_torch import MnM_MVP_torchrl
import unittest
from tensordict import TensorDictBase

class TestMnMMVPTorchRL(unittest.TestCase):
    def setUp(self):
        # Dummy company data
        self.dummy_company_data = pd.DataFrame({
            'fiscalDateEnding': ['2020-01-01', '2020-04-01'],
            'totalAssets': [1000.0, 1100.0],
            'shortTermDebt': [100.0, 120.0],
            'longTermDebt': [200.0, 210.0],
            'totalShareholderEquity': [700.0, 770.0],
            'totalRevenue': [500.0, 550.0],
            'costofGoodsAndServicesSold': [200.0, 220.0],
            'operatingExpenses': [100.0, 110.0],
            'interestIncome': [10.0, 12.0],
            'interestExpense': [15.0, 16.0],
            'depreciationAndAmortization': [20.0, 22.0],
            'incomeTaxExpense': [30.0, 33.0],
            'incomeBeforeTax': [100.0, 110.0],
            'ebit': [80.0, 88.0]
        })
        # Dummy macro simulator
        self.dummy_macro_simulator = MagicMock()
        self.dummy_macro_simulator.reset = MagicMock()
        self.dummy_macro_simulator.retrieve_origin_data = MagicMock(return_value=MagicMock(
            EFFR=pd.Series([0.02]),
            MARKET_RETURN=pd.Series([0.08])
        ))
        self.dummy_macro_simulator.retrieve_next_forecast = MagicMock(return_value=pd.DataFrame({
            'EFFR': [2.0],
            'MARKET_RETURN': [8.0],
            'REAL_GROWTH': [2.0],
            'dummy': [0.0]
        }))
        self.dummy_macro_simulator.default_output_columns = ['EFFR', 'MARKET_RETURN', 'REAL_GROWTH', 'dummy']
        self.dummy_calculate_beta = MagicMock(return_value=1.2)

        patcher1 = patch('environments.mnm_torch.calculate_beta', self.dummy_calculate_beta)
        patcher2 = patch('pandas.read_csv', MagicMock(return_value=self.dummy_company_data))
        patcher3 = patch('numpy.random.choice', MagicMock(return_value='2020-04-01'))
        patcher4 = patch('environments.mnm_torch.Dynamic_Simulation', MagicMock(return_value=self.dummy_macro_simulator))
        self.addCleanup(patcher1.stop)
        self.addCleanup(patcher2.stop)
        self.addCleanup(patcher3.stop)
        self.addCleanup(patcher4.stop)
        patcher1.start()
        patcher2.start()
        patcher3.start()
        patcher4.start()

        self.env = MnM_MVP_torchrl(
            ticker='DUMMY',
            fred_api_key='dummy',
            simulation_horizon=2,
            simulation_earliest_origin=pd.to_datetime('2020-01-01'),
            csv_path='dummy.csv',
            debug=True
        )
        self.env.macro_simulator = self.dummy_macro_simulator
        self.env.company_data = None
        # self.env.batch_size = torch.Size([1])  # Removed to avoid shape mismatch error
        self.env.to("cpu")

    def test_reset_returns_tensordict(self):
        result = self.env._reset({})
        self.assertIsInstance(result, TensorDictBase)
        self.assertIn("financial_metrics", result)
        self.assertIn("macro_economic_data", result)
        self.assertIn("done", result)
        self.assertIn("reward", result)

    def test_reset_financial_metrics_shape(self):
        result = self.env._reset({})
        fm = result["financial_metrics"]
        self.assertIsInstance(fm, torch.Tensor)
        self.assertEqual(fm.shape, (11,))

    def test_reset_macro_economic_data_shape(self):
        result = self.env._reset({})
        macro = result["macro_economic_data"]
        self.assertIsInstance(macro, torch.Tensor)
        self.assertEqual(macro.shape[0], len(self.env.macro_simulator.default_output_columns))

    def test_reset_done_and_reward(self):
        result = self.env._reset({})
        done = result["done"]
        reward = result["reward"]
        self.assertTrue(torch.equal(done, torch.tensor([[False]])))
        self.assertTrue(torch.equal(reward, torch.tensor([[0.0]], dtype=torch.float32)))

    def test_reset_sets_initial_state(self):
        self.env._reset({})
        self.assertEqual(self.env.total_asset, 1100.0)
        self.assertEqual(self.env.debt_st, 120.0)
        self.assertEqual(self.env.debt_lt, 210.0)
        self.assertEqual(self.env.equity, 770.0)
        self.assertEqual(self.env.total_asset_lag, 1000.0)
        self.assertEqual(self.env.debt_st_lag, 100.0)
        self.assertEqual(self.env.debt_lt_lag, 200.0)
        self.assertEqual(self.env.equity_lag, 700.0)
        self.assertEqual(self.env.revenue, 550.0)
        self.assertAlmostEqual(self.env.cogs_ratio, 220.0 / 550.0)
        self.assertAlmostEqual(self.env.opex_ratio, 110.0 / 550.0)
        self.assertEqual(self.env.interest_income, 12.0)
        self.assertAlmostEqual(self.env.interest_income_ratio, 12.0 / 1000.0)
        self.assertEqual(self.env.interest_expense, 16.0)
        self.assertEqual(self.env.depreciation_amortization, 22.0)
        self.assertAlmostEqual(self.env.tax_rate, 33.0 / 110.0)
        self.assertEqual(self.env.ebit, 88.0)

    def test_step_returns_tensordict(self):
        self.env._reset({})
        action = torch.tensor([1, 1], dtype=torch.long)
        tensordict = {"action": action}
        result = self.env._step(tensordict)
        self.assertIsInstance(result, TensorDictBase)
        self.assertIn("next", result)
        self.assertIn("reward", result)
        self.assertIn("done", result)
        self.assertIn("financial_metrics", result["next"])
        self.assertIn("macro_economic_data", result["next"])

    def test_step_bankruptcy_penalty(self):
        self.env._reset({})
        # Force equity to negative to trigger bankruptcy after step
        # Set total_asset and debts so equity will be negative after update
        self.env.total_asset = 100.0
        self.env.debt_st = 200.0
        self.env.debt_lt = 200.0
        action = torch.tensor([1, 1], dtype=torch.long)
        tensordict = {"action": action}
        result = self.env._step(tensordict)
        self.assertTrue(torch.equal(result["done"], torch.tensor([[True]])))
        self.assertTrue(torch.equal(result["reward"], torch.tensor([[-100.0]], dtype=torch.float32)))
        self.assertLess(self.env.equity, 0)

    def test_step_terminal_reward(self):
        self.env._reset({})
        self.env.current_step = self.env.horizon - 1
        action = torch.tensor([1, 1], dtype=torch.long)
        tensordict = {"action": action}
        result = self.env._step(tensordict)
        self.assertTrue(torch.equal(result["done"], torch.tensor([[True]])))
        # Reward should not be -100 (bankruptcy), so check it's not that
        self.assertFalse(torch.equal(result["reward"], torch.tensor([[-100.0]], dtype=torch.float32)))

    def test_revenue_model(self):
        self.env.revenue = 100.0
        new_revenue = self.env.revenue_model(0.05)
        self.assertAlmostEqual(new_revenue, 105.0)

    def test_cogs_model(self):
        self.env.revenue = 200.0
        self.env.cogs_ratio = 0.4
        cogs = self.env.cogs_model()
        self.assertAlmostEqual(cogs, 80.0)

    def test_opex_model(self):
        self.env.revenue = 300.0
        self.env.opex_ratio = 0.2
        opex = self.env.opex_model()
        self.assertAlmostEqual(opex, 60.0)

    def test_interest_income_model(self):
        self.env.total_asset_lag = 500.0
        self.env.interest_income_ratio = 0.01
        ii = self.env.interest_income_model()
        self.assertAlmostEqual(ii, 5.0)

    def test_interest_expense_model(self):
        self.env.debt_lt_lag = 100.0
        self.env.lt_interest_expense_ratio = 0.05
        self.env.debt_st_lag = 50.0
        self.env.st_interest_expense_ratio = 0.02
        ie = self.env.interest_expense_model()
        self.assertAlmostEqual(ie, 100.0 * 0.05 + 50.0 * 0.02)

    def test_depreciation_model(self):
        self.env.depreciation_amortization = 12.0
        self.assertEqual(self.env.depreciation_model(), 12.0)

    def test_st_debt_decay_model(self):
        self.env.debt_st = 120.0
        repayment = self.env.st_debt_decay_model(maturity=4)
        self.assertAlmostEqual(repayment, 120.0 / 4 / 4)
        self.assertAlmostEqual(self.env.debt_st, 120.0 - repayment)

    def test_lt_debt_convert_st_model(self):
        self.env.debt_lt = 200.0
        self.env.debt_st = 50.0
        self.env.lt_debt_convert_st_model(maturity=8)
        expected_repayment = 200.0 / 8 / 4
        self.assertAlmostEqual(self.env.debt_lt, 200.0 - expected_repayment)
        self.assertAlmostEqual(self.env.debt_st, 50.0 + expected_repayment)

if __name__ == "__main__":
    unittest.main()

import unittest
import pandas as pd
import numpy as np
from utils.interest_expense_to_debt_predictor import InterestExpensePredictor

class TestInterestExpensePredictor(unittest.TestCase):
    def setUp(self):
        # Create sample macro data with more rows
        dates = pd.date_range(start='2020-01-01', periods=30, freq='QE').strftime('%Y-%m-%d')
        np.random.seed(0)  # For reproducibility
        macro_data = {
            'date': dates,
            'EMPLOYMENT': np.random.uniform(90, 110, 30),
            'FPI': np.random.uniform(190, 210, 30),
            'GDP': np.random.uniform(990, 1020, 30),
            'GDP_DEF': np.random.uniform(100, 120, 30),
            'M2': np.random.uniform(490, 520, 30),
            'PCE': np.random.uniform(290, 310, 30),
            'PCE_PRICE_INDEX': np.random.uniform(110, 130, 30)
        }
        macro_df = pd.DataFrame(macro_data)
        macro_df.set_index('date', inplace=True)
        macro_df.to_csv('tests/test_macro.csv')

        # Create sample fundamental data with correlation to macro variables
        # Introduce correlation: interestExpense depends on GDP and M2
        interest_expense = 50 + 0.01 * (macro_df['GDP'] - 1000) + 0.02 * (macro_df['M2'] - 500) + np.random.normal(0, 2, 30)
        fund_data = {
            'Unnamed: 0': range(30),  # Dummy column to simulate structure
            'date': dates,
            'totalLiabilities': np.random.uniform(1000, 1200, 30),
            'interestExpense': interest_expense
        }
        fund_df = pd.DataFrame(fund_data)
        fund_df.to_csv('tests/test_fund.csv', index=False)

        self.predictor = InterestExpensePredictor(
            macro_file='tests/test_macro.csv',
            fund_file='tests/test_fund.csv'
        )

    def test_initialization(self):
        """Test if the predictor initializes correctly."""
        self.assertIsNotNone(self.predictor.model)
        self.assertIsNotNone(self.predictor.scaler)
        self.assertIsNotNone(self.predictor.new_scaler)
        self.assertIsNotNone(self.predictor.selected_features)
        self.assertIsNotNone(self.predictor.X)
        self.assertIsNotNone(self.predictor.y)

    def test_predict_on_training_data(self):
        """Test predictions on training data."""
        predictions = self.predictor.predict()
        self.assertEqual(len(predictions), len(self.predictor.y))
        self.assertTrue(all(isinstance(x, (float, np.floating)) for x in predictions))

    def test_predict_on_new_data(self):
        """Test predictions on new data with correct features."""
        selected_features = self.predictor.get_selected_features()
        new_data = pd.DataFrame(
            np.random.rand(2, len(selected_features)),
            columns=selected_features
        )
        predictions = self.predictor.predict(new_data)
        self.assertEqual(len(predictions), 2)
        self.assertTrue(all(isinstance(x, (float, np.floating)) for x in predictions))

    def test_predict_invalid_columns(self):
        """Test prediction with incorrect feature columns."""
        wrong_data = pd.DataFrame({
            'wrong_column': [1, 2]
        })
        with self.assertRaises(ValueError):
            self.predictor.predict(wrong_data)

    def test_get_selected_features(self):
        """Test get_selected_features method."""
        features = self.predictor.get_selected_features()
        self.assertIsInstance(features, list)
        self.assertTrue(len(features) > 0)
        self.assertTrue(all(isinstance(f, str) for f in features))

if __name__ == '__main__':
    unittest.main()
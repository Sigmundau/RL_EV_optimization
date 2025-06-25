import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

class InterestExpensePredictor:
    def __init__(self, macro_file="data/macro.csv", fund_file="data/KO_quarterly.csv"):
        self.macro_file = macro_file
        self.fund_file = fund_file
        self.model = None
        self.scaler = None
        self.new_scaler = None
        self.selected_features = None
        self.X = None
        self.y = None
        self._load_and_preprocess_data()
        self._fit_model()

    def _load_and_preprocess_data(self):
        # Load and preprocess macro data
        macro = pd.read_csv(self.macro_file, index_col=0)
        macro_non_stationary_cols = ['EMPLOYMENT', 'FPI', 'GDP', 'GDP_DEF', 'M2', 'PCE', 'PCE_PRICE_INDEX']
        for col in macro_non_stationary_cols:
            macro[f'{col}_pct_change'] = macro.loc[:, col].pct_change()
        macro.drop(columns=macro_non_stationary_cols, inplace=True)

        # Load and preprocess fundamental data
        ko = pd.read_csv(self.fund_file, index_col=1)
        if "Unnamed: 0" in ko.columns:
            ko.drop(columns="Unnamed: 0", inplace=True)
        ko = ko[::-1]
        ko = ko.loc[:, ['totalLiabilities', 'interestExpense']]
        ko['totalLiabilitiesPctChange'] = ko.loc[:, 'totalLiabilities'].pct_change()
        ko['interestExpensePctChange'] = ko.loc[:, 'interestExpense'].pct_change()
        ko['interestExpenseToDebtRatio'] = ko.loc[:, 'interestExpense'] / ko.loc[:, 'totalLiabilities']
        ko['fwdInterestExpenseToDebtRatio'] = ko['interestExpenseToDebtRatio'].shift(-1)
        ko.drop(columns=['totalLiabilities', 'interestExpense'], inplace=True)

        # Merge and clean data
        Xy = pd.merge(macro, ko, left_index=True, right_index=True, how='outer')
        Xy.dropna(inplace=True)
        self.X = Xy.drop(columns='fwdInterestExpenseToDebtRatio')
        self.y = Xy['fwdInterestExpenseToDebtRatio']

    def _fit_model(self):
        # Define parameters for TimeSeriesSplit
        n_samples = self.X.shape[0]
        min_train_size = 20
        test_size = 5
        n_splits = (n_samples - min_train_size) // test_size

        # Create expanding window cross-validator
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=0)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X)

        # Fit LassoCV
        lasso = LassoCV(
            alphas=np.logspace(-4, 0, 100),
            cv=tscv,
            random_state=0,
            n_jobs=-1
        )
        lasso.fit(X_scaled, self.y)

        # Store selected features
        coef = lasso.coef_
        selected_mask = coef != 0
        self.selected_features = self.X.columns[selected_mask]

        # Refit OLS on selected features
        new_X = self.X.loc[:, self.selected_features]
        self.new_scaler = StandardScaler()
        new_X_scaled = self.new_scaler.fit_transform(new_X)
        self.model = LinearRegression().fit(new_X_scaled, self.y)

    def predict(self, new_data=None):
        """
        Predict fwdInterestExpenseToDebtRatio.
        If new_data is None, predict on the training data.
        Otherwise, new_data should be a DataFrame with exactly the same columns as selected_features.
        """
        if new_data is None:
            data = self.X.loc[:, self.selected_features]
        else:
            # Validate new_data columns
            new_columns = set(new_data.columns)
            expected_columns = set(self.selected_features)
            
            missing_cols = expected_columns - new_columns
            if missing_cols:
                raise ValueError(f"Missing required features: {missing_cols}")
            
            extra_cols = new_columns - expected_columns
            if extra_cols:
                raise ValueError(f"Unexpected extra features: {extra_cols}")
            
            data = new_data.loc[:, self.selected_features]

        # Scale the data
        data_scaled = self.new_scaler.transform(data)

        # Make predictions
        predictions = self.model.predict(data_scaled)
        return predictions

    def get_selected_features(self):
        """Return the features selected by Lasso."""
        return list(self.selected_features)
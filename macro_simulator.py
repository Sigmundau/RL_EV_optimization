import pandas as pd
import numpy as np
from fredapi import Fred
import yfinance as yf
from functools import reduce
from statsmodels.tsa.api import VAR
# from statsmodels.tsa.statespace.varmax import VARMAX # Not yet used, but for VARMA consideration
from statsmodels.tools.eval_measures import rmse # For potential evaluation, not directly used in simulation but good to have

class Simulation:
    def __init__(self, fred_api_key: str, var_order: int = 1, debug: bool = False):
        """
        Initializes the Simulation class for joint macroeconomic forecasting.

        Args:
            fred_api_key (str): Your FRED API key.
            var_order (int): The order (number of lags) for the VAR model.
        """
        self.fred = Fred(api_key=fred_api_key)
        self.var_order = var_order
        self.model_fit = None
        self.simulated_path = None
        self.current_index = 0
        self.processed_data = None # Store the data prepared for VAR model

        # Define the columns that should be returned by retrieve_next_forecast
        self.default_output_columns = [
            'REAL_GROWTH', 'FX_STRENGTH', 'INFLATION', '2Y',
            '10Y','AAA_2Y','AAA_10Y','BAA_2Y','BAA_10Y'
        ]
        self.debug = debug
        # Fetch, prepare data, and fit VAR model
        self._fetch_and_prepare_data()
        self._fit_var_model()
        self.reset() # Generate an initial randomized path

    def _fetch_and_prepare_data(self):
        """
        Fetches historical data for the required macroeconomic variables from FRED
        and Yahoo Finance, and prepares them for VAR modeling at a quarterly resolution.
        All fetched and derived series are retained for the VAR model.
        """
        if self.debug:
            print("Fetching data from FRED and Yahoo Finance for joint simulation (Quarterly Resolution)...")

        # FRED Data - Combined list of all desired series
        fred_data_sources = {
            'AAAFF': 'AAAFF',
            'AAA10YM' : 'AAA10YM',
            'BAAFF': 'BAAFF',
            'BAA10YM' : 'BAA10YM',
            'EMPLOYMENT': 'CE16OV',
            'EFFR': 'FEDFUNDS',
            'FPI': 'FPI',
            'GDP': 'GDP',
            'GDP_DEF': 'GDPDEF',
            '10Y': 'GS10',      # Long Term Yield
            'M2': 'M2SL',
            'PCE': 'PCEC',
            'UNEMPLOYMENT': 'UNRATE',
            'PCE_PRICE_INDEX': 'PCEPI', # For inflation calculation
            '2Y': 'GS2',        # Short Term Yield
            'USD_INDEX': 'TWEXB' # Broad Trade Weighted US Dollar Index
        }

        all_fred_dataframes = []
        for col_name, series_id in fred_data_sources.items():
            df = pd.DataFrame(self.fred.get_series(series_id), columns=[col_name])
            all_fred_dataframes.append(df)

        # Merge FRED data using an OUTER merge for broader date range
        if not all_fred_dataframes:
            raise RuntimeError("No FRED dataframes to merge.")

        merged_df = all_fred_dataframes[0]
        for df in all_fred_dataframes[1:]:
            merged_df = pd.merge(merged_df, df, left_index=True, right_index=True, how='outer')


        # Data Pre-processing for VAR
        # 1. Ensure all data is numeric and handle NaNs at the original frequency first
        merged_df = merged_df.apply(pd.to_numeric, errors='coerce')
        merged_df.ffill(inplace=True) # Forward-fill any NaNs
        merged_df.bfill(inplace=True) # Backward-fill remaining NaNs (e.g., at the start)
        merged_df.dropna(inplace=True) # Drop any rows that are still all NaN after filling

        # 2. Resample all series to a consistent quarterly frequency ('QS' - Quarter Start)
        # Taking the last observation of the quarter for all series
        merged_df_quarterly = merged_df.resample('QS').last()
        
        # NOTE: No additional ffill/bfill here on merged_df_quarterly.
        # This is because we filled NaNs on the higher frequency merged_df first.

        # All processed series (original and derived) are retained for the VAR model
        # Drop any remaining NaNs after all transformations.
        self.processed_data = merged_df_quarterly.dropna()

        if self.processed_data.empty:
            raise ValueError("Processed data for VAR model is empty after transformations. Check data availability and transformations.")

        if self.debug:
            print("Data fetching and preparation complete.")
            print(f"Prepared data shape for VAR (all variables): {self.processed_data.shape}")
            print(f"Columns included in VAR model: {self.processed_data.columns.tolist()}") # Double check print
            print(self.processed_data.tail())


    def _fit_var_model(self):
        """
        Fits a VAR model to the prepared macroeconomic data.
        """
        if self.processed_data is None or self.processed_data.empty:
            raise RuntimeError("Processed data not available or empty. Call _fetch_and_prepare_data first.")

        if self.debug:
            print(f"Fitting VAR({self.var_order}) model with {self.processed_data.shape[1]} endogenous variables...")
            print("WARNING: A VAR model with many variables and limited data can lead to overfitting and unstable results.")
        try:
            model = VAR(self.processed_data)
            self.model_fit = model.fit(self.var_order)
            if self.debug:
                print("VAR model fitted successfully.")
                # print(self.model_fit.summary()) # Uncomment to see summary (might be very long)
        except Exception as e:
            if self.debug:
                print(f"Error fitting VAR model: {e}")
            raise

    def reset(self, path_length: int = 100, noise_scale: float = 1.0, random_start_window_quarters: int = 40):
        """
        Resets the simulation by generating a new randomized path for all variables,
        starting from a randomly selected quarter within the historical data.

        Args:
            path_length (int): The desired length of the new simulated path.
            noise_scale (float): A multiplier for the standard deviation of the residuals,
                                 controlling the randomness/volatility of the new path.
            random_start_window_quarters (int): The number of recent quarters from the end
                                                 of the historical data from which a random
                                                 start date can be chosen.
        """
        if self.model_fit is None:
            raise RuntimeError("VAR model has not been fitted. Call _fit_var_model() first.")
        if self.processed_data is None or self.processed_data.empty:
            raise RuntimeError("Processed data not available. Cannot reset simulation.")

        if self.debug:
            print(f"Resetting joint simulation with a new randomized path of length {path_length} (Quarterly Resolution)...")

        min_start_idx = self.var_order + 1
        max_start_idx = len(self.processed_data) - 1

        start_window_limit_idx = len(self.processed_data) - random_start_window_quarters - 1
        max_start_idx = min(max_start_idx, len(self.processed_data) - 1)
        
        effective_min_idx = max(min_start_idx, start_window_limit_idx)

        if effective_min_idx >= max_start_idx:
            if self.debug:
                print(f"Warning: random_start_window_quarters ({random_start_window_quarters}) is too large or data is too short. "
                  f"Selecting random start from available range [{self.processed_data.index[min_start_idx].to_period('Q')}] to [{self.processed_data.index[max_start_idx].to_period('Q')}]")
            random_start_idx = np.random.randint(min_start_idx, max_start_idx + 1)
        else:
            random_start_idx = np.random.randint(effective_min_idx, max_start_idx + 1)

        initial_conditions = self.processed_data.iloc[random_start_idx - self.var_order : random_start_idx].values
        sim_path_start_date = self.processed_data.index[random_start_idx]

        params = self.model_fit.params
        sigma_u = self.model_fit.sigma_u

        num_variables = self.processed_data.shape[1]

        shocks = np.random.multivariate_normal(
            np.zeros(num_variables),
            sigma_u * noise_scale,
            size=path_length
        )

        simulated_data = initial_conditions.tolist()

        for i in range(path_length):
            predicted_next_val = self.model_fit.forecast(y=np.array(simulated_data[-self.var_order:]), steps=1)[0]
            next_val_with_shock = predicted_next_val + shocks[i]
            simulated_data.append(next_val_with_shock.tolist())

        final_simulated_path = np.array(simulated_data[self.var_order:])

        simulated_index = pd.to_datetime(pd.date_range(start=sim_path_start_date, periods=path_length, freq='QS'))

        self.simulated_path = pd.DataFrame(
            final_simulated_path,
            index=simulated_index,
            columns=self.processed_data.columns # All variables are in the simulated path
        )
        
        # 3. Calculate derived variables. These will also become part of the VAR model.
        # Real Growth: Quarterly GDP Growth (Quarter-over-Quarter percentage change)
        self.simulated_path['REAL_GROWTH'] = self.simulated_path['GDP_REAL      '].pct_change(periods=1) * 100

        # FX Strength: Quarterly change in USD Index
        self.simulated_path['FX_STRENGTH'] = self.simulated_path['USD_INDEX'].pct_change(periods=1) * 100

        # Expected Inflation: Quarterly PCE Price Index growth (Quarter-over-Quarter percentage change)
        self.simulated_path['INFLATION'] = self.simulated_path['PCE_PRICE_INDEX'].pct_change(periods=1) * 100

        self.simulated_path['AAA_10Y'] = self.simulated_path['10Y'] + self.simulated_path['AAA10YM'] 
        self.simulated_path['BAA_10Y'] = self.simulated_path['10Y'] + self.simulated_path['BAA10YM'] 
        #Interpolate (linear) 2Y spread
        self.simulated_path['AAA_2Y'] = self.simulated_path['2Y'] + self.simulated_path['AAAFF'] + (self.simulated_path['AAA10YM'] - self.simulated_path['AAAFF'])*0.2
        self.simulated_path['BAA_2Y'] = self.simulated_path['2Y'] + self.simulated_path['BAAFF'] + (self.simulated_path['BAA10YM'] - self.simulated_path['BAAFF'])*0.2

        self.simulated_path.dropna(inplace=True)
        self.current_index = 0
        if self.debug:
            print(f"New randomized joint path generated. Start Date: {sim_path_start_date.to_period('Q')}")
            print(f"Simulated path starts from: {self.simulated_path.index[0].to_period('Q')}")
            print(f"Simulated path ends at: {self.simulated_path.index[-1].to_period('Q')}")


    def retrieve_next_forecast(self, steps: int = 1) -> pd.DataFrame:
        """
        Retrieves the next 'steps' forecasted values for a selected subset of variables
        from the current position in the simulated path.

        Args:
            steps (int): The number of future values to retrieve.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the next 'steps'
                          forecasted values for the selected variables.
        """
        if self.simulated_path is None:
            raise RuntimeError("No simulated path available. Call reset() first.")

        if self.current_index >= len(self.simulated_path):
            if self.debug:
                print(f"End of simulated path reached. Generating new path...")
            self.reset() # Automatically reset if path exhausted

        end_index = self.current_index + steps
        
        # Select only the default output columns before returning
        forecasted_values = self.simulated_path.iloc[self.current_index:min(end_index, len(self.simulated_path))][self.default_output_columns]
        self.current_index = min(end_index, len(self.simulated_path))

        return forecasted_values


# --- Example Usage ---
# if __name__ == "__main__":
#     FRED_API_KEY = "5446f1dec140788657af4c0720a225b2" # Your FRED API key from the notebook

#     print("--- Joint Macroeconomic Simulation (Quarterly Resolution with Random Start - All Data in VAR) ---")
#     try:
#         macro_simulation = Simulation(
#             fred_api_key=FRED_API_KEY,
#             var_order=4 # Example: using 4 lags for quarterly data
#         )

#         print("\nInitial Joint Forecast (5 steps / 5 quarters - Selected Outputs):")
#         forecast_initial = macro_simulation.retrieve_next_forecast(steps=5)
#         print(forecast_initial)

#         print("\nAnother 3 steps (3 quarters) of Joint Forecast (Selected Outputs):")
#         forecast_next = macro_simulation.retrieve_next_forecast(steps=3)
#         print(forecast_next)

#         print("\nResetting Joint Simulation and forecasting 10 steps (10 quarters) with custom window (Selected Outputs):")
#         macro_simulation.reset(path_length=20, noise_scale=1.2, random_start_window_quarters=20)
#         forecast_reset = macro_simulation.retrieve_next_forecast(steps=10)
#         print(forecast_reset)

#         print("\nContinuously retrieving until path end (and auto-reset) (Selected Outputs):")
#         for _ in range(3):
#             next_vals = macro_simulation.retrieve_next_forecast(steps=5)
#             print(f"\nNext 5 values (from {next_vals.index[0].to_period('Q')} to {next_vals.index[-1].to_period('Q')}): \n{next_vals.to_string()}")

#     except Exception as e:
#         print(f"An error occurred during joint macroeconomic simulation: {e}")
#         print("Please ensure your FRED API key is correct and the FRED series IDs are valid and have sufficient data.")
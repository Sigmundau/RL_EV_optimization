import pandas as pd
import numpy as np
from fredapi import Fred
import yfinance as yf
from functools import reduce
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse # For potential evaluation, not directly used in simulation but good to have

class Simulation:
    def __init__(self, fred_api_key: str, var_order: int = 1,debug: bool = False):
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
        self.debug = debug

        # Fetch, prepare data, and fit VAR model
        self._fetch_and_prepare_data()
        self._fit_var_model()
        self.reset() # Generate an initial randomized path

    def _fetch_and_prepare_data(self):
        """
        Fetches historical data for the required macroeconomic variables from FRED
        and Yahoo Finance, and prepares them for VAR modeling at a quarterly resolution.
        """
        if self.debug:
            print("Fetching data from FRED and Yahoo Finance for joint simulation (Quarterly Resolution)...")

        # FRED Data - Combined list of all desired series
        fred_data_sources = {
            'AAA': 'AAA',
            'BAA': 'BAA',
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
        # Any NaNs here would mean no data even after filling, which will be handled by final dropna.


        # 3. Calculate growth rates and inflation for stationarity (if needed for VAR)
        # All calculations will now be based on quarterly data.

        # Real Growth: Quarterly GDP Growth (Quarter-over-Quarter percentage change)
        merged_df_quarterly['REAL_GROWTH'] = merged_df_quarterly['GDP'].pct_change(periods=1) * 100

        # FX Strength: Quarterly change in USD Index
        merged_df_quarterly['FX_STRENGTH'] = merged_df_quarterly['USD_INDEX'].pct_change(periods=1) * 100

        # Expected Inflation: Quarterly PCE Price Index growth (Quarter-over-Quarter percentage change)
        merged_df_quarterly['INFLATION'] = merged_df_quarterly['PCE_PRICE_INDEX'].pct_change(periods=1) * 100

        # Yields: Using levels for now. These are typically monthly, so taking the last monthly value for the quarter.
        merged_df_quarterly['SHORT_TERM_YIELD'] = merged_df_quarterly['2Y']
        merged_df_quarterly['LONG_TERM_YIELD'] = merged_df_quarterly['10Y']

        # Select only the target variables for the VAR model and drop any remaining NaNs
        # (e.g., from the first row of pct_change)
        self.processed_data = merged_df_quarterly[[
            'REAL_GROWTH', 'FX_STRENGTH', 'INFLATION', 'SHORT_TERM_YIELD', 'LONG_TERM_YIELD'
        ]].dropna()

        if self.processed_data.empty:
            raise ValueError("Processed data for VAR model is empty after transformations. Check data availability and transformations.")
        if self.debug:
            print("Data fetching and preparation complete.")
            print(f"Prepared data shape for VAR: {self.processed_data.shape}")
            print(self.processed_data.tail())


    def _fit_var_model(self):
        """
        Fits a VAR model to the prepared macroeconomic data.
        """
        if self.processed_data is None or self.processed_data.empty:
            raise RuntimeError("Processed data not available or empty. Call _fetch_and_prepare_data first.")
        if self.debug:
            print(f"Fitting VAR({self.var_order}) model...")
        try:
            # Statsmodels VAR requires observations as rows and variables as columns
            # Our self.processed_data is already in this format.
            model = VAR(self.processed_data)
            self.model_fit = model.fit(self.var_order)
            if self.debug:            
                print("VAR model fitted successfully.")
            # print(self.model_fit.summary()) # Uncomment to see summary
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

        # Determine the valid range for the random start date
        # The start date needs to have at least 'var_order' preceding observations
        # to form the initial conditions for the VAR forecast.
        min_start_idx = self.var_order
        # Ensure max_start_idx leaves enough room for var_order lags *before* the start date
        max_start_idx = len(self.processed_data) - 1

        # Limit the max_start_idx to the random_start_window_quarters
        start_window_limit_idx = len(self.processed_data) - random_start_window_quarters - 1
        max_start_idx = min(max_start_idx, len(self.processed_data) - 1) # Ensure max_start_idx is within bounds
        
        # Define the actual usable range for random_start_idx
        # It must be at least var_order, and within the random_start_window_quarters from the end.
        effective_min_idx = max(min_start_idx, start_window_limit_idx)

        if effective_min_idx >= max_start_idx:
            if self.debug:
                print(f"Warning: random_start_window_quarters ({random_start_window_quarters}) is too large or data is too short. "
                      f"Selecting random start from available range [{self.processed_data.index[min_start_idx].to_period('Q')}] to [{self.processed_data.index[max_start_idx].to_period('Q')}]")
            random_start_idx = np.random.randint(min_start_idx, max_start_idx + 1)
        else:
            # Randomly select an index within the determined window
            random_start_idx = np.random.randint(effective_min_idx, max_start_idx + 1)

        # Get the initial conditions from the actual data ending at the randomly chosen start index
        initial_conditions = self.processed_data.iloc[random_start_idx - self.var_order : random_start_idx].values

        # Get the actual start date for the simulated path
        sim_path_start_date = self.processed_data.index[random_start_idx]

        # Get the coefficients and covariance matrix of residuals from the fitted VAR model
        params = self.model_fit.params # VAR coefficients
        sigma_u = self.model_fit.sigma_u # Covariance matrix of residuals

        num_variables = self.processed_data.shape[1]

        # Generate multivariate normal noise for the simulation
        shocks = np.random.multivariate_normal(
            np.zeros(num_variables),
            sigma_u * noise_scale,
            size=path_length
        )

        simulated_data = initial_conditions.tolist() # Start with actual data as initial conditions

        # Simulate the path
        for i in range(path_length):
            predicted_next_val = self.model_fit.forecast(y=np.array(simulated_data[-self.var_order:]), steps=1)[0]
            next_val_with_shock = predicted_next_val + shocks[i]
            simulated_data.append(next_val_with_shock.tolist())

        # Discard the initial_conditions used for warm-up and only keep the simulated path
        final_simulated_path = np.array(simulated_data[self.var_order:])

        # Create a proper date index for the simulated path starting from the random date
        simulated_index = pd.to_datetime(pd.date_range(start=sim_path_start_date, periods=path_length, freq='QS'))

        self.simulated_path = pd.DataFrame(
            final_simulated_path,
            index=simulated_index,
            columns=self.processed_data.columns
        )
        self.current_index = 0
        if self.debug:
            print(f"New randomized joint path generated. Start Date: {sim_path_start_date.to_period('Q')}")
            print(f"Simulated path starts from: {self.simulated_path.index[0].to_period('Q')}")
            print(f"Simulated path ends at: {self.simulated_path.index[-1].to_period('Q')}")


    def retrieve_next_forecast(self, steps: int = 1) -> pd.DataFrame:
        """
        Retrieves the next 'steps' forecasted values for all variables
        from the current position in the simulated path.

        Args:
            steps (int): The number of future values to retrieve.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the next 'steps'
                          forecasted values for all variables.
        """
        if self.simulated_path is None:
            raise RuntimeError("No simulated path available. Call reset() first.")

        if self.current_index >= len(self.simulated_path):
            if self.debug:
                print(f"End of simulated path reached. Generating new path...")
            self.reset() # Automatically reset if path exhausted

        end_index = self.current_index + steps
        forecasted_values = self.simulated_path.iloc[self.current_index:min(end_index, len(self.simulated_path))]
        self.current_index = min(end_index, len(self.simulated_path))

        return forecasted_values

# # --- Example Usage ---
# if __name__ == "__main__":
#     FRED_API_KEY = "5446f1dec140788657af4c0720a225b2" # Your FRED API key from the notebook

#     print("--- Joint Macroeconomic Simulation (Quarterly Resolution with Random Start) ---")
#     try:
#         # Initialize the simulation with a VAR(2) model (2 lags)
#         # Using a higher var_order might be appropriate for quarterly data if patterns are longer-term
#         macro_simulation = Simulation(
#             fred_api_key=FRED_API_KEY,
#             var_order=4 # Example: using 4 lags for quarterly data (equivalent to 1 year)
#         )

#         print("\nInitial Joint Forecast (5 steps / 5 quarters):")
#         forecast_initial = macro_simulation.retrieve_next_forecast(steps=5)
#         print(forecast_initial)

#         print("\nAnother 3 steps (3 quarters) of Joint Forecast:")
#         forecast_next = macro_simulation.retrieve_next_forecast(steps=3)
#         print(forecast_next)

#         print("\nResetting Joint Simulation and forecasting 10 steps (10 quarters) with custom window:")
#         # random_start_window_quarters=20 means the start date will be randomly chosen from the last 20 quarters
#         # (5 years) of historical data, ensuring enough preceding data for VAR lags.
#         macro_simulation.reset(path_length=20, noise_scale=1.2, random_start_window_quarters=20)
#         forecast_reset = macro_simulation.retrieve_next_forecast(steps=10)
#         print(forecast_reset)

#         print("\nContinuously retrieving until path end (and auto-reset):")
#         for _ in range(3): # Retrieve a few times to show auto-reset
#             next_vals = macro_simulation.retrieve_next_forecast(steps=5)
#             print(f"\nNext 5 values (from {next_vals.index[0].to_period('Q')} to {next_vals.index[-1].to_period('Q')}): \n{next_vals.to_string()}")

#     except Exception as e:
#         print(f"An error occurred during joint macroeconomic simulation: {e}")
#         print("Please ensure your FRED API key is correct and the FRED series IDs are valid and have sufficient data.")
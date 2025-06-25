import pandas as pd
import numpy as np
from fredapi import Fred
import yfinance as yf
from functools import reduce
from statsmodels.tsa.api import VAR
# from statsmodels.tsa.statespace.varmax import VARMAX # Not yet used, but for VARMA consideration
from arch import arch_model
from statsmodels.tools.eval_measures import \
    rmse  # For potential evaluation, not directly used in simulation but good to have
import datetime


class Simulation:
    def __init__(self, fred_api_key: str, var_order: int = 1, debug: bool = False, fred_csv_path: str = "data/FRED_data.csv"):
        """
        Initializes the Simulation class for joint macroeconomic forecasting.

        Args:
            fred_api_key (str): Your FRED API key.
            var_order (int): The order (number of lags) for the VAR model.
            debug (bool): If True, enables debug output for tracing computation values.
            csv_path (str): Path to a CSV file containing historical data for the macroeconomic variables.
        """
        self.fred = Fred(fred_api_key)
        self.var_order = var_order
        self.model_fit = None
        self.simulated_path = None
        self.current_index = 0
        self.processed_data = None  # Store the data prepared for VAR model
        self.fred_csv_path = fred_csv_path  # Path to CSV file for historical data, if provided

        # Define the columns that should be returned by retrieve_next_forecast
        self.default_output_columns = [
            'REAL_GROWTH', 'FX_STRENGTH', 'INFLATION','EFFR','2Y',
            '10Y', 'AAA_2Y', 'AAA_10Y', 'BAA_2Y', 'BAA_10Y', 'MARKET_RETURN'
        ]
        self.debug = debug
        # Fetch, prepare data, and fit VAR model
        self._fetch_and_prepare_data()
        self._fit_var_model()
        self.reset()  # Generate an initial randomized path

    def _fetch_and_prepare_data(self):
        """
        Fetches historical data for the required macroeconomic variables from FRED
        and Yahoo Finance, and prepares them for VAR modeling at a quarterly resolution.
        All fetched and derived series are retained for the VAR model.
        """
        if self.debug:
            print("Fetching data from FRED and Yahoo Finance for joint simulation (Quarterly Resolution)...")

        if self.fred_csv_path != "":
            self.processed_data = pd.read_csv(self.fred_csv_path, index_col=0)
            self.processed_data.index = pd.to_datetime(self.processed_data.index)
        else:
            # FRED Data - Combined list of all desired series
            fred_data_sources = {
                'AAAFF': 'AAAFF',
                'AAA10YM': 'AAA10YM',
                'BAAFF': 'BAAFF',
                'BAA10YM': 'BAA10YM',
                'EMPLOYMENT': 'CE16OV',
                'EFFR': 'FEDFUNDS',
                'FPI': 'FPI',
                'GDP': 'GDP',
                'GDP_DEF': 'GDPDEF',
                '10Y': 'GS10',  # Long Term Yield
                'M2': 'M2SL',
                'PCE': 'PCEC',
                'UNEMPLOYMENT': 'UNRATE',
                'PCE_PRICE_INDEX': 'PCEPI',  # For inflation calculation
                '2Y': 'GS2',  # Short Term Yield
                'USD_INDEX': 'TWEXB'  # Broad Trade Weighted US Dollar Index
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
            merged_df.ffill(inplace=True)  # Forward-fill any NaNs
            merged_df.bfill(inplace=True)  # Backward-fill remaining NaNs (e.g., at the start)
            merged_df.dropna(inplace=True)  # Drop any rows that are still all NaN after filling

            # 2. Resample all series to a consistent quarterly frequency ('QS' - Quarter Start)
            # Taking the last observation of the quarter for all series
            merged_df_quarterly = merged_df.resample('QE').last()

            # 3. Suppress yield into actual value, also convert into spread format
            # BBB = AAA + BBB_AAA_spread ; AAA = EFFR+ AAA_spread

            merged_df_quarterly['AAA_SPREAD_FF'] = merged_df_quarterly['AAAFF'] - merged_df_quarterly['EFFR']
            merged_df_quarterly['AAA_SPREAD_10Y'] = merged_df_quarterly['AAA10YM'] - merged_df_quarterly['EFFR']
            merged_df_quarterly['BAA_AAA_SPREAD_FF'] = merged_df_quarterly['BAAFF'] - merged_df_quarterly['AAAFF']
            merged_df_quarterly['BAA_AAA_SPREAD_10Y'] = merged_df_quarterly['BAA10YM'] - merged_df_quarterly['AAA10YM']

            # 4. Identify variables that should be log-transformed to ensure no negative curve
            log_transform_vars = ['EFFR', '10Y', '2Y']
            for col in log_transform_vars:
                if col in merged_df_quarterly.columns:
                    merged_df_quarterly[col] = np.log(merged_df_quarterly[col])

            # 5. add market return
            spx = yf.Ticker('^GSPC')
            spx_data = spx.history(period='max')['Close'].resample('QE').last()
            spx_data.index = spx_data.index.tz_localize(None)
            new_df = pd.DataFrame(spx_data.pct_change())
            new_df.dropna(inplace=True)
            new_df.columns = ['MARKET_RETURN']

            merged_df_quarterly = pd.merge(merged_df_quarterly, new_df, left_index=True, right_index=True, how='left')

            # All processed series (original and derived) are retained for the VAR model
            # Drop any remaining NaNs after all transformations.
            # Exclude the original absolute AAA/BAA yields, include the spreads instead
            var_columns = [
                'EMPLOYMENT', 'EFFR', 'FPI', 'GDP', 'GDP_DEF', '10Y', 'M2', 'PCE',
                'UNEMPLOYMENT', 'PCE_PRICE_INDEX', '2Y', 'USD_INDEX', 'MARKET_RETURN',
                'AAA_SPREAD_FF', 'AAA_SPREAD_10Y', 'BAA_AAA_SPREAD_FF', 'BAA_AAA_SPREAD_10Y'
            ]
            self.processed_data = merged_df_quarterly[var_columns].dropna()

            if self.processed_data.empty:
                raise ValueError(
                    "Processed data for VAR model is empty after transformations. Check data availability and transformations.")

            if self.debug:
                print("Data fetching and preparation complete.")
                print(f"Prepared data shape for VAR (all variables): {self.processed_data.shape}")
                print(f"Columns included in VAR model: {self.processed_data.columns.tolist()}")  # Double check print
                print(self.processed_data.tail())

    def _fit_var_model(self):
        """
        Fits a VAR model to the prepared macroeconomic data and then fits
        univariate GARCH models to the VAR residuals.
        """
        if self.processed_data is None or self.processed_data.empty:
            raise RuntimeError("Processed data not available or empty. Call _fetch_and_prepare_data first.")

        if self.debug:
            print(f"Fitting VAR({self.var_order}) model with {self.processed_data.shape[1]} endogenous variables...")

        try:
            model = VAR(self.processed_data)
            self.model_fit = model.fit(self.var_order)
            if self.debug:
                print("VAR model fitted successfully.")
            # print(self.model_fit.summary()) # Uncomment to see summary

            # --- NEW: Extract VAR Residuals and Fit Univariate GARCH Models ---
            self.var_residuals = self.model_fit.resid  # Get residuals from the VAR model
            self.garch_models = {}  # To store fitted GARCH models for each residual series
            self.initial_garch_forecast_variance = {}  # To store the last conditional variance for simulation start

            if self.debug:
                print(f"Fitting univariate GARCH(1,1) models to VAR residuals...")

            for col_name in self.var_residuals.columns:
                try:
                    residual_series = self.var_residuals[col_name].copy()
                    current_std = residual_series.std()
                    target_std = 100.0  # Aim for a standard deviation around 100 (can be adjusted)

                    scaling_factor = 1.0
                    if current_std > 1e-10: # Avoid division by zero for near-zero std
                        scaling_factor = target_std / current_std
                    
                    # Optional: Add a safety cap/floor to the scaling factor if it becomes extreme
                    # For example, if scaling_factor is ridiculously large, it might cause overflow later.
                    # This also ensures we don't scale by factors like 0.00000000001
                    if scaling_factor < 1e-5: # Don't shrink too much
                        scaling_factor = 1e-5
                    elif scaling_factor > 1e15: # Don't expand too much
                         scaling_factor = 1e15

                    rescaled_residual_series = residual_series * scaling_factor
                    am = arch_model(rescaled_residual_series, vol='Garch', p=1, q=1, dist='ged',rescale=False)
                    res = am.fit(update_freq=0, disp='off')  # disp='off' to suppress solver output
                    self.garch_models[col_name] = res

                    # Store the last conditional variance to start the simulation
                    # This will be the starting point for forecasting volatility in reset()
                    if not res.conditional_variance.empty:
                        self.initial_garch_forecast_variance[col_name] = res.conditional_variance.iloc[-1]
                    else:
                        # Fallback if no conditional variance is estimated (e.g., constant variance)
                        self.initial_garch_forecast_variance[col_name] = res.resid.var()

                    if self.debug:
                        print(f"  - Fitted GARCH for {col_name}. Log-likelihood: {res.loglikelihood:.4f}")
                except Exception as garch_e:
                    if self.debug:
                        print(f"  - Warning: Could not fit GARCH for {col_name}: {garch_e}. Using constant variance.")
                    # Fallback to constant variance if GARCH fails
                    self.garch_models[col_name] = None
                    self.initial_garch_forecast_variance[col_name] = self.var_residuals[col_name].var()

        except Exception as e:
            if self.debug:
                print(f"Error fitting VAR model: {e}")
            raise

    def reset(self, path_length: int = 100, noise_scale: float = 1.0, random_start_window_quarters: int = 40,
              origin: pd.Timestamp = None):
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
        if self.model_fit is None or not self.garch_models:
            raise RuntimeError("VAR or GARCH models have not been fitted. Call _fit_var_model() first.")
        if self.processed_data is None or self.processed_data.empty:
            raise RuntimeError("Processed data not available. Cannot reset simulation.")

        if self.debug:
            print(
                f"Resetting joint simulation with a new randomized path of length {path_length} (Quarterly Resolution)...")

        if origin is not None:  # Changed to 'is not None' for clarity
            # Find the integer position of the origin timestamp
            if origin is not None:
            # Ensure origin is a pandas Timestamp. If it's a datetime.date, convert it.
                if isinstance(origin, datetime.date) and not isinstance(origin, datetime.datetime):
                    origin = pd.Timestamp(origin)

            # Find the closest quarter-end date in your processed_data index
            # This is more robust than exact get_loc for user-provided dates
            try:
                # Use asof to find the last valid observation up to and including the origin
                # Or use BDay/BDay to find the exact business day if required
                aligned_origin = self.processed_data.index[self.processed_data.index <= origin].max()
                if pd.isna(aligned_origin):
                    raise ValueError(f"No data found on or before the origin timestamp {origin}.")
                random_start_idx = self.processed_data.index.get_loc(aligned_origin)
                
            except Exception as e:
                raise ValueError(f"Could not find a suitable start date for origin {origin}: {e}")

            # Ensure there are enough preceding observations for the VAR order
            if random_start_idx < self.var_order:
                raise ValueError(
                    f"Origin timestamp {origin} is too early. Need at least {self.var_order} observations before it for VAR model.")
        else:
            min_start_idx = self.var_order + 1
            max_start_idx = len(self.processed_data) - 1

            start_window_limit_idx = len(self.processed_data) - random_start_window_quarters - 1
            max_start_idx = min(max_start_idx, len(self.processed_data) - 1)

            effective_min_idx = max(min_start_idx, start_window_limit_idx)

            if effective_min_idx >= max_start_idx:
                if self.debug:
                    print(
                        f"Warning: random_start_window_quarters ({random_start_window_quarters}) is too large or data is too short. "
                        f"Selecting random start from available range [{self.processed_data.index[min_start_idx].to_period('Q')}] to [{self.processed_data.index[max_start_idx].to_period('Q')}]")
                random_start_idx = np.random.randint(min_start_idx, max_start_idx + 1)
            else:
                random_start_idx = np.random.randint(effective_min_idx, max_start_idx + 1)

        # Now random_start_idx is an integer index
        initial_conditions = self.processed_data.iloc[random_start_idx - self.var_order: random_start_idx].values
        sim_path_start_date = self.processed_data.index[random_start_idx]

        params = self.model_fit.params
        num_variables = self.processed_data.shape[1]

        simulated_data = initial_conditions.tolist()
        current_garch_variances = np.array([
            self.initial_garch_forecast_variance.get(col_name, self.var_residuals[col_name].var())
            for col_name in self.processed_data.columns
        ])

        if self.debug:
            print(f"Initial GARCH variances for simulation: {current_garch_variances}")

        for i in range(path_length):
            forecast_input = np.array(simulated_data[-self.var_order:])
            predicted_next_val = self.model_fit.forecast(y=forecast_input, steps=1)[0]

            # --- NEW: Generate GARCH-informed shocks ---
            # Generate independent standard normal shocks
            standard_normal_shocks = np.random.normal(0, 1, size=num_variables)

            # Apply conditional standard deviations to the shocks
            # We assume a diagonal covariance matrix here (i.e., no dynamic correlation)
            # This implements the "simple GARCH" part.
            conditional_stds = np.sqrt(current_garch_variances)
            garch_shocks = standard_normal_shocks * conditional_stds * noise_scale

            # Add the shocks to the forecasted mean
            next_val_with_shock = predicted_next_val + garch_shocks.flatten()  # .flatten() to ensure correct shape

            simulated_data.append(next_val_with_shock.tolist())

            # --- NEW: Update GARCH conditional variances for the *next* step based on the *current* simulated shock ---
            # This is crucial for volatility clustering.
            # You need the actual residual from the current simulated step to update the GARCH recursion.
            # Since we don't have the "true" residual (only the simulated shock), we use the shock as proxy.
            # This is a simplification; a full GARCH simulation would recursively apply the GARCH equation.
            # A more robust approach might involve a custom GARCH simulation function that takes previous shocks.
            for j, col_name in enumerate(self.processed_data.columns):
                garch_model_res = self.garch_models.get(col_name)
                if garch_model_res:
                    # Access GARCH parameters for update: omega, alpha, beta
                    # This requires knowing the parameter order from arch_model.summary()
                    # For GARCH(1,1) with dist='normal', params are often: mu, omega, alpha[1], beta[1]
                    # This is a simplified manual GARCH(1,1) update.
                    # A more reliable way is to use arch's forecast method if it supports recursive updates easily.

                    # Example GARCH(1,1) equation for variance: var_t = omega + alpha * resid_{t-1}^2 + beta * var_{t-1}
                    omega = garch_model_res.params['omega']
                    alpha1 = garch_model_res.params['alpha[1]']
                    beta1 = garch_model_res.params['beta[1]']

                    # Use the squared *simulated shock* as the "previous residual squared"
                    # This is a key approximation in simulation.
                    squared_simulated_shock = garch_shocks[j] ** 2

                    # Update the conditional variance for the next step
                    next_conditional_variance = omega + alpha1 * squared_simulated_shock + beta1 * \
                                                current_garch_variances[j]
                    current_garch_variances[j] = next_conditional_variance
                else:
                    # If no GARCH model was fitted, variance remains constant (historical variance)
                    pass

        final_simulated_path = np.array(simulated_data[self.var_order:])

        simulated_index = pd.to_datetime(pd.date_range(start=sim_path_start_date, periods=path_length, freq='QE'))

        self.simulated_path = pd.DataFrame(
            final_simulated_path,
            index=simulated_index,
            columns=self.processed_data.columns  # All variables are in the simulated path
        )
        # Inverse Log transformation for non-negative values
        log_transform_vars = ['EFFR', '10Y', '2Y']
        for col in log_transform_vars:
            if col in self.simulated_path.columns:
                self.simulated_path[col] = np.exp(np.clip(self.simulated_path[col], np.log(0.0001), np.log(0.5)))

        # Reconstitute absolute yield values from simulated spreads
        self.simulated_path['AAA_FF'] = self.simulated_path['EFFR'] + self.simulated_path['AAA_SPREAD_FF']
        self.simulated_path['AAA_10Y'] = self.simulated_path['EFFR'] + self.simulated_path['AAA_SPREAD_10Y']
        self.simulated_path['BAA_FF'] = self.simulated_path['AAA_FF'] + self.simulated_path['BAA_AAA_SPREAD_FF']
        self.simulated_path['BAA_10Y'] = self.simulated_path['AAA_10Y'] + self.simulated_path['BAA_AAA_SPREAD_10Y']

        # Calculate derived variables
        self.simulated_path['REAL_GROWTH'] = self.simulated_path['GDP_DEF'].pct_change(periods=1) * 100
        self.simulated_path['FX_STRENGTH'] = self.simulated_path['USD_INDEX'].pct_change(periods=1) * 100
        self.simulated_path['INFLATION'] = self.simulated_path['PCE_PRICE_INDEX'].pct_change(periods=1) * 100
        self.simulated_path['AAA_2Y'] = self.simulated_path['2Y'] + self.simulated_path['AAA_FF'] + (
                self.simulated_path['AAA_10Y'] - self.simulated_path['AAA_FF']) * 0.2
        self.simulated_path['BAA_2Y'] = self.simulated_path['2Y'] + self.simulated_path['BAA_FF'] + (
                self.simulated_path['BAA_10Y'] - self.simulated_path['BAA_FF']) * 0.2

        self.simulated_path.dropna(inplace=True)
        self.current_index = 1  # Start from the second row since the first is the initial condition
        if self.debug:
            print(f"New randomized joint path generated. Start Date: {sim_path_start_date.to_period('Q')}")
            print(f"Simulated path shape: {self.simulated_path.shape}")
            print(f"Simulated path starts from: {self.simulated_path.index[0].to_period('Q')}")
            print(f"Simulated path ends at: {self.simulated_path.index[-1].to_period('Q')}")

    def retrieve_origin_data(self) -> pd.DataFrame:
        """
        Retrieves the origin of the forecasting values.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the origin of forecasts.
        """
        if self.simulated_path is None:
            raise RuntimeError("Processed data not available. Call _fetch_and_prepare_data() first.")
        return self.simulated_path.iloc[0].to_frame().T

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
            self.reset()  # Automatically reset if path exhausted

        end_index = self.current_index + steps

        # Select only the default output columns before returning
        forecasted_values = self.simulated_path.iloc[self.current_index:min(end_index, len(self.simulated_path))][
            self.default_output_columns]
        self.current_index = min(end_index, len(self.simulated_path))

        return forecasted_values

class Dynamic_Simulation:
    def __init__(self, fred_api_key: str, var_order: int = None, debug: bool = False, fred_csv_path: str = "data/FRED_data.csv"):
        """
        Initializes the Simulation class for joint macroeconomic forecasting.
        
        Args:
            fred_api_key (str): Your FRED API key.
            var_order (int): The order (number of lags) for the VAR model. If None, will be selected automatically.
            debug (bool): If True, enables debug output for tracing computation values.
            fred_csv_path (str): Path to a CSV file containing historical data for the macroeconomic variables.
        """
        self.fred = Fred(fred_api_key)
        self.var_order = var_order  # Can be None for automatic selection
        self.model_fit = None
        self.simulated_path = None
        self.current_index = 0
        self.processed_data = None # Store the data prepared for VAR model
        self.fred_csv_path = fred_csv_path # Path to CSV file for historical data, if provided
        
        # Define the columns that should be returned by retrieve_next_forecast
        self.default_output_columns = [
            'REAL_GROWTH', 'FX_STRENGTH', 'INFLATION','EFFR','2Y',
            '10Y', 'AAA_2Y', 'AAA_10Y', 'BAA_2Y', 'BAA_10Y', 'MARKET_RETURN'
        ]
        
        self.debug = debug
        
        # Storage for GARCH model specifications
        self.garch_specs = {}
        
        # Fetch, prepare data, and fit VAR model
        self._fetch_and_prepare_data()
        self._fit_enhanced_var_garch_model()
        self.reset() # Generate an initial randomized path

    def _fetch_and_prepare_data(self):
        """
        Fetches historical data for the required macroeconomic variables from FRED
        and Yahoo Finance, and prepares them for VAR modeling at a quarterly resolution.
        All fetched and derived series are retained for the VAR model.
        """
        if self.debug:
            print("Fetching data from FRED and Yahoo Finance for joint simulation (Quarterly Resolution)...")
            
        if self.fred_csv_path != "":
            self.processed_data = pd.read_csv(self.fred_csv_path, index_col=0)
            self.processed_data.index = pd.to_datetime(self.processed_data.index)
        else:
            # FRED Data - Combined list of all desired series
            fred_data_sources = {
                'AAAFF': 'AAAFF',
                'AAA10YM': 'AAA10YM',
                'BAAFF': 'BAAFF',
                'BAA10YM': 'BAA10YM',
                'EMPLOYMENT': 'CE16OV',
                'EFFR': 'FEDFUNDS',
                'FPI': 'FPI',
                'GDP': 'GDP',
                'GDP_DEF': 'GDPDEF',
                '10Y': 'GS10', # Long Term Yield
                'M2': 'M2SL',
                'PCE': 'PCEC',
                'UNEMPLOYMENT': 'UNRATE',
                'PCE_PRICE_INDEX': 'PCEPI', # For inflation calculation
                '2Y': 'GS2', # Short Term Yield
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
            merged_df_quarterly = merged_df.resample('QE').last()
            
            # 3. Suppress yield into actual value, also convert into spread format
            # BBB = AAA + BBB_AAA_spread ; AAA = EFFR+ AAA_spread
            merged_df_quarterly['AAA_SPREAD_FF'] = merged_df_quarterly['AAAFF'] - merged_df_quarterly['EFFR']
            merged_df_quarterly['AAA_SPREAD_10Y'] = merged_df_quarterly['AAA10YM'] - merged_df_quarterly['EFFR']
            merged_df_quarterly['BAA_AAA_SPREAD_FF'] = merged_df_quarterly['BAAFF'] - merged_df_quarterly['AAAFF']
            merged_df_quarterly['BAA_AAA_SPREAD_10Y'] = merged_df_quarterly['BAA10YM'] - merged_df_quarterly['AAA10YM']
            
            # 4. Identify variables that should be log-transformed to ensure no negative curve
            log_transform_vars = ['EFFR', '10Y', '2Y']
            for col in log_transform_vars:
                if col in merged_df_quarterly.columns:
                    merged_df_quarterly[col] = np.log(merged_df_quarterly[col])
            
            # 5. add market return
            spx = yf.Ticker('^GSPC')
            spx_data = spx.history(period='max')['Close'].resample('QE').last()
            spx_data.index = spx_data.index.tz_localize(None)
            new_df = pd.DataFrame(spx_data.pct_change())
            new_df.dropna(inplace=True)
            new_df.columns = ['MARKET_RETURN']
            merged_df_quarterly = pd.merge(merged_df_quarterly, new_df, left_index=True, right_index=True, how='left')
            
            # All processed series (original and derived) are retained for the VAR model
            # Drop any remaining NaNs after all transformations.
            # Exclude the original absolute AAA/BAA yields, include the spreads instead
            var_columns = [
                'EMPLOYMENT', 'EFFR', 'FPI', 'GDP', 'GDP_DEF', '10Y', 'M2', 'PCE',
                'UNEMPLOYMENT', 'PCE_PRICE_INDEX', '2Y', 'USD_INDEX', 'MARKET_RETURN',
                'AAA_SPREAD_FF', 'AAA_SPREAD_10Y', 'BAA_AAA_SPREAD_FF', 'BAA_AAA_SPREAD_10Y'
            ]
            
            self.processed_data = merged_df_quarterly[var_columns].dropna()
            
        if self.processed_data.empty:
            raise ValueError("Processed data for VAR model is empty after transformations. Check data availability and transformations.")
        
        if self.debug:
            print("Data fetching and preparation complete.")
            print(f"Prepared data shape for VAR (all variables): {self.processed_data.shape}")
            print(f"Columns included in VAR model: {self.processed_data.columns.tolist()}")
            print(self.processed_data.tail())

    def _select_optimal_var_order(self, max_order):
        """Select optimal VAR order using information criteria with sample size adjustments."""
        results = []
        
        for order in range(1, max_order + 1):
            try:
                # Check degrees of freedom
                required_obs = order + 1
                available_obs = len(self.processed_data)
                
                if available_obs <= required_obs:
                    continue
                    
                model = VAR(self.processed_data)
                fitted = model.fit(order)
                
                # Use sample-size adjusted criteria
                n = len(fitted.resid)
                k = self.processed_data.shape[1]
                
                # Enhanced BIC with small sample correction
                log_det_sigma = np.log(np.linalg.det(fitted.sigma_u))
                aic = log_det_sigma + (2 * order * k**2) / n
                bic = log_det_sigma + (np.log(n) * order * k**2) / n
                
                # Small sample AIC correction (AICc)
                aicc = aic + (2 * order * k**2 * (order * k**2 + 1)) / (n - order * k**2 - 1)
                
                results.append({
                    'order': order,
                    'aic': aic,
                    'bic': bic,
                    'aicc': aicc,
                    'fitted': fitted
                })
                
                if self.debug:
                    print(f"VAR({order}): AIC={aic:.4f}, BIC={bic:.4f}, AICc={aicc:.4f}")
                
            except Exception as e:
                if self.debug:
                    print(f"Failed to fit VAR({order}): {e}")
                continue
        
        if not results:
            return 1  # Fallback to VAR(1)
        
        # Select based on BIC (more conservative for small samples)
        best = min(results, key=lambda x: x['bic'])
        return best['order']

    def _select_optimal_garch_enhanced(self, residual_series, col_name):
        """
        Enhanced GARCH selection with small sample considerations and multiple criteria.
        """
        current_std = residual_series.std()
        if current_std < 1e-6:
            if self.debug:
                print(f" - Skipping GARCH selection for {col_name}: insufficient variation")
            return None, "insufficient_variation"
        
        # Scale residuals for numerical stability
        target_std = 100.0
        scaling_factor = target_std / current_std if current_std > 1e-10 else 1.0
        scaling_factor = np.clip(scaling_factor, 1e-5, 1e5)
        rescaled_residual_series = residual_series * scaling_factor
        
        # Conservative candidate set for small samples
        candidate_specs = [
            # Start with simple models
            {'vol': 'ConstantVariance', 'dist': 'normal'},
            {'vol': 'Arch', 'p': 1, 'dist': 'normal'},
            {'vol': 'Garch', 'p': 1, 'q': 1, 'dist': 'normal'},
            {'vol': 'Garch', 'p': 1, 'q': 1, 'dist': 'ged'},
            {'vol': 'Garch', 'p': 1, 'q': 1, 'dist': 't'},
        ]
        
        # Only try more complex models if we have enough observations
        if len(residual_series) > 30:
            candidate_specs.extend([
                {'vol': 'Garch', 'p': 2, 'q': 1, 'dist': 'normal'},
                {'vol': 'Garch', 'p': 1, 'q': 2, 'dist': 'normal'},
                {'vol': 'EGARCH', 'p': 1, 'q': 1, 'dist': 'normal'},
                {'vol': 'EGARCH', 'p': 1, 'q': 1, 'dist': 'ged'},
            ])
        
        best_model = None
        best_bic = np.inf
        best_spec_info = None
        
        for spec in candidate_specs:
            try:
                # Suppress convergence warnings for cleaner output
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=Warning)
                    
                    am = arch_model(rescaled_residual_series, rescale=False, **spec)
                    res = am.fit(update_freq=0, disp='off', options={'maxiter': 1000})
                
                # Check convergence
                if hasattr(res, 'convergence_flag') and res.convergence_flag != 0:
                    continue
                
                # Use BIC for small samples (more conservative)
                current_bic = res.bic
                
                if current_bic < best_bic:
                    best_bic = current_bic
                    best_model = res
                    best_spec_info = spec.copy()
                    
                if self.debug:
                    print(f" - {col_name}: {spec} -> BIC: {current_bic:.4f}")
                    
            except Exception as e:
                if self.debug:
                    print(f" - Failed {spec} for {col_name}: {e}")
                continue
        
        if best_model is None:
            if self.debug:
                print(f" - No GARCH model converged for {col_name}. Using constant variance.")
            return None, "no_convergence"
        
        if self.debug:
            print(f" - Best model for {col_name}: {best_spec_info} (BIC: {best_bic:.4f})")
        
        return best_model, best_spec_info

    def _fit_enhanced_var_garch_model(self):
        """
        Enhanced VAR-GARCH with dynamic model selection and robustness checks.
        """
        if self.processed_data is None or self.processed_data.empty:
            raise RuntimeError("Processed data not available.")

        # Step 1: Enhanced VAR order selection
        if self.var_order is None:
            # Conservative rule for small samples: max order = T / (3 * k)
            max_var_order = min(4, int(len(self.processed_data) / (self.processed_data.shape[1] * 3)))
            max_var_order = max(1, max_var_order)  # Ensure at least order 1
            
            self.var_order = self._select_optimal_var_order(max_var_order)
            
            if self.debug:
                print(f"Auto-selected VAR order: {self.var_order}")
        else:
            if self.debug:
                print(f"Using pre-specified VAR order: {self.var_order}")
        
        # Step 2: Fit VAR model
        try:
            model = VAR(self.processed_data)
            self.model_fit = model.fit(self.var_order)
            
            if self.debug:
                print("VAR model fitted successfully.")
                
        except Exception as e:
            if self.debug:
                print(f"Error fitting VAR model: {e}")
            raise
        
        # Step 3: Enhanced GARCH fitting with dynamic selection
        self.var_residuals = self.model_fit.resid
        self.garch_models = {}
        self.garch_specs = {}  # Store selected specifications
        self.initial_garch_forecast_variance = {}
        
        if self.debug:
            print(f"Fitting GARCH models to VAR residuals using dynamic selection...")
        
        for col_name in self.var_residuals.columns:
            try:
                residual_series = self.var_residuals[col_name].copy()
                
                garch_model, spec_info = self._select_optimal_garch_enhanced(residual_series, col_name)
                
                self.garch_models[col_name] = garch_model
                self.garch_specs[col_name] = spec_info
                
                if garch_model is not None:
                    if not garch_model.conditional_variance.empty:
                        self.initial_garch_forecast_variance[col_name] = garch_model.conditional_variance.iloc[-1]
                    else:
                        self.initial_garch_forecast_variance[col_name] = garch_model.resid.var()
                else:
                    self.initial_garch_forecast_variance[col_name] = self.var_residuals[col_name].var()
                    
            except Exception as garch_e:
                if self.debug:
                    print(f" - Error in GARCH selection for {col_name}: {garch_e}. Using constant variance.")
                self.garch_models[col_name] = None
                self.garch_specs[col_name] = "constant_variance"
                self.initial_garch_forecast_variance[col_name] = self.var_residuals[col_name].var()
        
        if self.debug:
            print("Enhanced VAR-GARCH model fitting complete.")
            print(f"GARCH specifications: {self.garch_specs}")

    # Keep the original method name for backward compatibility
    def _fit_var_model(self):
        """Backward compatibility wrapper for the enhanced method."""
        self._fit_enhanced_var_garch_model()

    def reset(self, path_length: int = 100, noise_scale: float = 1.0, random_start_window_quarters: int = 40,
              origin: pd.Timestamp = None):
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
        if self.model_fit is None or not self.garch_models:
            raise RuntimeError("VAR or GARCH models have not been fitted. Call _fit_var_model() first.")
        
        if self.processed_data is None or self.processed_data.empty:
            raise RuntimeError("Processed data not available. Cannot reset simulation.")
        
        if self.debug:
            print(f"Resetting joint simulation with a new randomized path of length {path_length} (Quarterly Resolution)...")

        if origin is not None:
            # Find the integer position of the origin timestamp
            if isinstance(origin, datetime.date) and not isinstance(origin, datetime.datetime):
                origin = pd.Timestamp(origin)
            
            try:
                aligned_origin = self.processed_data.index[self.processed_data.index <= origin].max()
                if pd.isna(aligned_origin):
                    raise ValueError(f"No data found on or before the origin timestamp {origin}.")
                random_start_idx = self.processed_data.index.get_loc(aligned_origin)
            except Exception as e:
                raise ValueError(f"Could not find a suitable start date for origin {origin}: {e}")
            
            if random_start_idx < self.var_order:
                raise ValueError(f"Origin timestamp {origin} is too early. Need at least {self.var_order} observations before it for VAR model.")
        else:
            min_start_idx = self.var_order + 1
            max_start_idx = len(self.processed_data) - 1
            start_window_limit_idx = len(self.processed_data) - random_start_window_quarters - 1
            
            effective_min_idx = max(min_start_idx, start_window_limit_idx)
            
            if effective_min_idx >= max_start_idx:
                if self.debug:
                    print(f"Warning: random_start_window_quarters ({random_start_window_quarters}) is too large or data is too short. "
                          f"Selecting random start from available range.")
                random_start_idx = np.random.randint(min_start_idx, max_start_idx + 1)
            else:
                random_start_idx = np.random.randint(effective_min_idx, max_start_idx + 1)

        # Initialize simulation
        initial_conditions = self.processed_data.iloc[random_start_idx - self.var_order: random_start_idx].values
        sim_path_start_date = self.processed_data.index[random_start_idx]
        
        params = self.model_fit.params
        num_variables = self.processed_data.shape[1]
        simulated_data = initial_conditions.tolist()
        
        # Initialize GARCH variances
        current_garch_variances = np.array([
            self.initial_garch_forecast_variance.get(col_name, self.var_residuals[col_name].var())
            for col_name in self.processed_data.columns
        ])
        
        if self.debug:
            print(f"Initial GARCH variances for simulation: {current_garch_variances}")

        # Main simulation loop
        for i in range(path_length):
            forecast_input = np.array(simulated_data[-self.var_order:])
            predicted_next_val = self.model_fit.forecast(y=forecast_input, steps=1)[0]
            
            # Generate GARCH-informed shocks
            standard_normal_shocks = np.random.normal(0, 1, size=num_variables)
            conditional_stds = np.sqrt(current_garch_variances)
            garch_shocks = standard_normal_shocks * conditional_stds * noise_scale
            
            # Add bounds to prevent extreme values during simulation
            for j, col_name in enumerate(self.processed_data.columns):
                if col_name in ['EFFR', '10Y', '2Y']:  # Log-transformed variables
                    garch_shocks[j] = np.clip(garch_shocks[j], -2, 2)  # Reasonable bounds for log rates
            
            next_val_with_shock = predicted_next_val + garch_shocks.flatten()
            simulated_data.append(next_val_with_shock.tolist())
            
            # Update GARCH conditional variances for next step
            for j, col_name in enumerate(self.processed_data.columns):
                garch_model_res = self.garch_models.get(col_name)
                if garch_model_res:
                    try:
                        # Manual GARCH(1,1) update
                        omega = garch_model_res.params['omega']
                        alpha1 = garch_model_res.params['alpha[1]']
                        beta1 = garch_model_res.params['beta[1]']
                        
                        squared_simulated_shock = garch_shocks[j] ** 2
                        next_conditional_variance = omega + alpha1 * squared_simulated_shock + beta1 * current_garch_variances[j]
                        
                        # Ensure positive variance
                        current_garch_variances[j] = max(1e-8, next_conditional_variance)
                    except:
                        # If parameter access fails, keep variance constant
                        pass

        # Create final simulated path
        final_simulated_path = np.array(simulated_data[self.var_order:])
        simulated_index = pd.to_datetime(pd.date_range(start=sim_path_start_date, periods=path_length, freq='QE'))
        
        self.simulated_path = pd.DataFrame(
            final_simulated_path,
            index=simulated_index,
            columns=self.processed_data.columns
        )
        
        # Inverse Log transformation for non-negative values with bounds
        log_transform_vars = ['EFFR', '10Y', '2Y']
        for col in log_transform_vars:
            if col in self.simulated_path.columns:
                self.simulated_path[col] = np.exp(np.clip(self.simulated_path[col], np.log(0.0001), np.log(0.5)))
        
        # Reconstitute absolute yield values from simulated spreads
        self.simulated_path['AAA_FF'] = self.simulated_path['EFFR'] + self.simulated_path['AAA_SPREAD_FF']
        self.simulated_path['AAA_10Y'] = self.simulated_path['EFFR'] + self.simulated_path['AAA_SPREAD_10Y']
        self.simulated_path['BAA_FF'] = self.simulated_path['AAA_FF'] + self.simulated_path['BAA_AAA_SPREAD_FF']
        self.simulated_path['BAA_10Y'] = self.simulated_path['AAA_10Y'] + self.simulated_path['BAA_AAA_SPREAD_10Y']
        
        # Calculate derived variables
        self.simulated_path['REAL_GROWTH'] = self.simulated_path['GDP_DEF'].pct_change(periods=1) * 100
        self.simulated_path['FX_STRENGTH'] = self.simulated_path['USD_INDEX'].pct_change(periods=1) * 100
        self.simulated_path['INFLATION'] = self.simulated_path['PCE_PRICE_INDEX'].pct_change(periods=1) * 100
        self.simulated_path['AAA_2Y'] = self.simulated_path['2Y'] + self.simulated_path['AAA_FF'] + (
            self.simulated_path['AAA_10Y'] - self.simulated_path['AAA_FF']) * 0.2
        self.simulated_path['BAA_2Y'] = self.simulated_path['2Y'] + self.simulated_path['BAA_FF'] + (
            self.simulated_path['BAA_10Y'] - self.simulated_path['BAA_FF']) * 0.2
        
        self.simulated_path.dropna(inplace=True)
        self.current_index = 1
        
        if self.debug:
            print(f"New randomized joint path generated. Start Date: {sim_path_start_date.to_period('Q')}")
            print(f"Simulated path shape: {self.simulated_path.shape}")
            print(f"Simulated path starts from: {self.simulated_path.index[0].to_period('Q')}")
            print(f"Simulated path ends at: {self.simulated_path.index[-1].to_period('Q')}")

    def retrieve_origin_data(self) -> pd.DataFrame:
        """
        Retrieves the origin of the forecasting values.
        
        Returns:
            pd.DataFrame: A pandas DataFrame containing the origin of forecasts.
        """
        if self.simulated_path is None:
            raise RuntimeError("Processed data not available. Call _fetch_and_prepare_data() first.")
        return self.simulated_path.iloc[0].to_frame().T

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
            self.reset()
        
        end_index = self.current_index + steps
        
        # Select only the default output columns before returning
        forecasted_values = self.simulated_path.iloc[self.current_index:min(end_index, len(self.simulated_path))][
            self.default_output_columns]
        
        self.current_index = min(end_index, len(self.simulated_path))
        
        return forecasted_values
    
    def get_model_summary(self):
        """
        Returns a summary of the fitted models.
        
        Returns:
            dict: Summary information about VAR and GARCH models.
        """
        summary = {
            'var_order': self.var_order,
            'data_shape': self.processed_data.shape if self.processed_data is not None else None,
            'garch_specs': self.garch_specs,
            'var_aic': getattr(self.model_fit, 'aic', None) if self.model_fit else None,
            'var_bic': getattr(self.model_fit, 'bic', None) if self.model_fit else None,
        }
        return summary

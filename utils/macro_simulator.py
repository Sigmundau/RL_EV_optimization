import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from fredapi import Fred
import yfinance as yf

class ECBPoolSimulator:
    """
    ECBPoolSimulator simulates macroeconomic variables using a combination of VAR and toy DSGE models.
    It fetches, processes, and simulates macroeconomic time series data, supporting both real-time data
    from FRED and local CSVs. The simulator can generate future macroeconomic scenarios for use in
    reinforcement learning or economic modeling.
    """
    def __init__(self, fred_api_key, var_order=4, debug=False,fixed_path = False, fred_csv_path="",fixed_path_csv=""):
        """
        Initialize the ECBPoolSimulator.
        
        Args:
            fred_api_key (str): API key for FRED data access.
            var_order (int): Order of the VAR model.
            debug (bool): If True, print debug information.
            fixed_path (bool): If True, return the same simulated path, used for benchmark comparison
            fred_csv_path (str): Optional path to a CSV file with pre-fetched FRED data.
            fixed_path_csv(str): Optional path to a CSV file with a pretrained simulation path for uniform purpose.
        """
        self.debug = debug
        self.fred = Fred(fred_api_key)
        self.var_order = var_order
        self.fred_csv_path = fred_csv_path
        self.processed_data = None
        self.differenced_data = None
        self.last_observed_level = None
        self.model_fit = None
        self.simulated_path = None
        self.fixed_path = fixed_path
        self.fixed_path_csv = fixed_path_csv
        self.current_index = 0
        self.default_output_columns = [
            'REAL_GROWTH', 'FX_STRENGTH', 'INFLATION','EFFR','2Y',
            '10Y', 'AAA_2Y', 'AAA_10Y', 'BAA_2Y', 'BAA_10Y', 'MARKET_RETURN',
            'EMPLOYMENT', 'PCE', 'GDP', 'FPI','AAA_SPREAD_FF'
        ]
        self._fetch_and_prepare_data()
        self._fit_var_model()
        self.reset()

    def _fetch_and_prepare_data(self):
        """
        Fetch and prepare macroeconomic data from FRED or a local CSV. Processes, merges, and transforms
        the data for use in the VAR and DSGE models. Handles log transforms, resampling, and derived columns.
        """
        if self.debug:
            print("Fetching and preparing data...")
        if self.fred_csv_path != "":
            self.processed_data = pd.read_csv(self.fred_csv_path, index_col=0)
            self.processed_data.index = pd.to_datetime(self.processed_data.index)
        else:
            fred_data_sources = {
                'AAA_SPREAD_FF': 'AAAFF',
                'AAA_SPREAD_10Y': 'AAA10YM',
                'BAA_SPREAD_FF': 'BAAFF',
                'BAA_SPREAD_10Y': 'BAA10YM',
                'EMPLOYMENT': 'CE16OV',
                'EFFR': 'FEDFUNDS',
                'FPI': 'FPI',
                'GDP': 'GDP',
                'GDP_DEF': 'GDPDEF',
                '10Y': 'GS10',
                'M2': 'M2SL',
                'PCE': 'PCEC',
                'UNEMPLOYMENT': 'UNRATE',
                'PCE_PRICE_INDEX': 'PCEPI',
                '2Y': 'GS2',
            }
            twexb = self.fred.get_series('TWEXB')
            dtwexbgs = self.fred.get_series('DTWEXBGS')
            overlap_start = max(twexb.index.min(), dtwexbgs.index.min())
            scale = dtwexbgs.loc[dtwexbgs.index==overlap_start] / twexb.loc[twexb.index==overlap_start]
            twexb_rebased = twexb * scale
            combined_usd_index = pd.concat([
                twexb_rebased.loc[:overlap_start - pd.Timedelta(days=1)],
                dtwexbgs.loc[overlap_start:]
            ])
            all_fred_dataframes = [pd.DataFrame(combined_usd_index, columns=['USD_INDEX'])]
            for col_name, series_id in fred_data_sources.items():
                df = pd.DataFrame(self.fred.get_series(series_id), columns=[col_name])
                all_fred_dataframes.append(df)
            merged_df = pd.concat(all_fred_dataframes, axis=1, join='outer')
            merged_df = merged_df.apply(pd.to_numeric, errors='coerce')
            merged_df.dropna(inplace=True)
            merged_df_quarterly = merged_df.resample('QE').last()
            merged_df_quarterly['BAA_AAA_SPREAD_FF'] = merged_df_quarterly['BAA_SPREAD_FF'] - merged_df_quarterly['AAA_SPREAD_FF']
            merged_df_quarterly['BAA_AAA_SPREAD_10Y'] = merged_df_quarterly['BAA_SPREAD_10Y'] - merged_df_quarterly['AAA_SPREAD_10Y']
            log_transform_vars = ['EFFR', '10Y', '2Y']
            for col in log_transform_vars:
                if col in merged_df_quarterly.columns:
                    merged_df_quarterly[col] = np.log(merged_df_quarterly[col])
            spx = yf.Ticker('^GSPC')
            spx_data = spx.history(period='max')['Close'].resample('QE').last()
            spx_data.index = spx_data.index.tz_localize(None)
            new_df = pd.DataFrame(spx_data.pct_change())
            new_df.dropna(inplace=True)
            new_df.columns = ['MARKET_RETURN']
            merged_df_quarterly = pd.merge(merged_df_quarterly, new_df, left_index=True, right_index=True, how='left')
            var_columns = [
                'EMPLOYMENT', 'EFFR', 'FPI', 'GDP', 'GDP_DEF', '10Y', 'M2', 'PCE',
                'UNEMPLOYMENT', 'PCE_PRICE_INDEX', '2Y', 'USD_INDEX', 'MARKET_RETURN',
                'AAA_SPREAD_FF', 'AAA_SPREAD_10Y', 'BAA_AAA_SPREAD_FF', 'BAA_AAA_SPREAD_10Y'
            ]
            self.processed_data = merged_df_quarterly[var_columns].dropna()
            self.last_observed_level = self.processed_data.iloc[-1].copy()
            self.differenced_data = self.processed_data.diff().dropna()
        if self.debug:
            print("Data ready. Shape:", self.processed_data.shape)

    def _fit_var_model(self):
        """
        Fit a VAR model to the differenced macroeconomic data.
        """
        self.var_model = VAR(self.differenced_data)
        self.model_fit = self.var_model.fit(self.var_order)
        if self.debug:
            print("VAR model fitted.")

    def _simulate_var(self, steps, n_draws, last_obs):
        """
        Simulate future macroeconomic paths using the fitted VAR model.
        
        Args:
            steps (int): Number of time steps to simulate.
            n_draws (int): Number of simulation draws.
            last_obs (np.ndarray): Initial values for the VAR simulation.
        Returns:
            np.ndarray: Simulated differenced values (n_draws, steps, n_vars).
        """
        results = []
        for draw in range(n_draws):
            sim = self.model_fit.simulate_var(steps=steps, initial_values=last_obs)
            results.append(sim)
        return np.array(results)  # shape: (n_draws, steps, n_vars)

    def _simulate_dsge(self, steps, n_draws, last_obs):
        """
        Simulate a toy DSGE (New Keynesian) block for GDP and inflation.
        
        Args:
            steps (int): Number of time steps to simulate.
            n_draws (int): Number of simulation draws.
            last_obs (np.ndarray): Initial values for the simulation.
        Returns:
            np.ndarray: Simulated differenced values (n_draws, steps, n_vars).
        """
        # Toy DSGE block: simple New Keynesian, only for GDP, PCE_PRICE_INDEX
        gdp_idx = self.differenced_data.columns.get_loc('GDP')
        pi_idx = self.differenced_data.columns.get_loc('PCE_PRICE_INDEX')
        # Use last observed differences as initial state
        last = last_obs[-1]
        sim_results = []
        for draw in range(n_draws):
            sim_path = []
            x, pi = last[gdp_idx], last[pi_idx]
            for t in range(steps):
                e1, e2, e3 = np.random.normal(0, 0.1, 3)
                x_new = 0.7*x + 0.2*pi + e1
                pi_new = 0.6*pi + 0.1*x + e2
                sim_step = np.zeros(len(self.differenced_data.columns))
                sim_step[gdp_idx] = x_new
                sim_step[pi_idx] = pi_new
                # sim_step[effr_idx] = i_new
                sim_path.append(sim_step)
                x, pi = x_new, pi_new
            sim_results.append(sim_path)
        return np.array(sim_results)  # shape: (n_draws, steps, n_vars)

    def reset(self, path_length=10, n_draws=100, pool_weights=None, 
              random_start_window_quarters=40, origin=None):
        """
        Reset the simulator and generate a new simulated macroeconomic path.
        
        Args:
            path_length (int): Number of forecast steps.
            n_draws (int): Number of simulation draws.
            pool_weights (dict): Weights for combining VAR and DSGE simulations.
            random_start_window_quarters (int): Window for randomizing simulation start.
            origin (str or pd.Timestamp): Optional fixed start date for simulation.
        """
        # for predetermined path
        if self.fixed_path_csv!="":
            self.simulated_path = pd.read_csv(self.fixed_path_csv,index_col=0)
            self.simulated_path.index = pd.to_datetime(self.simulated_path.index)
        # regenerate path if not fixed path or initial path for fixed path
        elif not self.fixed_path or (self.simulated_path is None):
            # Include the origin data in the path
            path_length +=1  
            
            if pool_weights is None:
                pool_weights = {'VAR': 0.5, 'DSGE': 0.5}
            # --- Choose simulation start ---
            if origin is not None:
                # Ensure origin is a pd.Timestamp for comparison with DatetimeIndex
                if isinstance(origin, str):
                    origin = pd.Timestamp(origin)
                elif hasattr(origin, 'year') and not isinstance(origin, pd.Timestamp):
                    # Handles datetime.date and datetime.datetime
                    origin = pd.Timestamp(origin)
                aligned_origin = self.processed_data.index[self.processed_data.index <= origin].max()
                if pd.isna(aligned_origin):
                    raise ValueError(f"No data found on or before the origin timestamp {origin}.")
                start_idx = self.processed_data.index.get_loc(aligned_origin)
            else:
                min_idx = self.var_order
                max_idx = len(self.processed_data) - 1
                start_window_limit = max(min_idx, max_idx - random_start_window_quarters)
                start_idx = np.random.randint(start_window_limit, max_idx + 1)
            # --- Initial conditions ---
            last_obs = self.differenced_data.iloc[start_idx - self.var_order:start_idx].values
            last_level = self.processed_data.iloc[start_idx - 1].values
            # --- Simulate ---
            var_draws = self._simulate_var(path_length, n_draws, last_obs=last_obs)
            dsge_draws = self._simulate_dsge(path_length, n_draws, last_obs=last_obs)
            pooled_draws = pool_weights['VAR'] * var_draws + pool_weights['DSGE'] * dsge_draws
            # --- Reconstruct levels ---
            levels = []
            for sim in pooled_draws:
                lv = [last_level]
                for diff in sim:
                    lv.append(lv[-1] + diff)
                levels.append(np.array(lv[1:]))
            levels = np.array(levels)  # (n_draws, steps, n_vars)
            mean_path = levels.mean(axis=0)
            simulated_index = pd.date_range(start=self.processed_data.index[start_idx], periods=path_length, freq='QE')
            self.simulated_path = pd.DataFrame(mean_path, index=simulated_index, columns=self.differenced_data.columns)
            # --- Inverse log for rates ---
            for col in ['EFFR', '10Y', '2Y']:
                if col in self.simulated_path.columns:
                    self.simulated_path[col] = np.exp(self.simulated_path[col])
            # --- Derived variables ---
            self.simulated_path['REAL_GROWTH'] = self.simulated_path['GDP_DEF'].pct_change(periods=1)
            self.simulated_path['FX_STRENGTH'] = self.simulated_path['USD_INDEX'].pct_change(periods=1)
            self.simulated_path['INFLATION'] = self.simulated_path['PCE_PRICE_INDEX'].pct_change(periods=1)
            self.simulated_path['AAA_2Y'] = self.simulated_path['2Y'] + self.simulated_path['AAA_SPREAD_FF'] + (self.simulated_path['AAA_SPREAD_10Y'] - self.simulated_path['AAA_SPREAD_FF']) * 0.2
            self.simulated_path['BAA_2Y'] =  self.simulated_path['AAA_2Y'] + self.simulated_path['BAA_AAA_SPREAD_FF'] + (self.simulated_path['BAA_AAA_SPREAD_10Y'] - self.simulated_path['BAA_AAA_SPREAD_FF']) * 0.2
            self.simulated_path['AAA_10Y'] = self.simulated_path['10Y'] + self.simulated_path['AAA_SPREAD_10Y']
            self.simulated_path['BAA_10Y'] = self.simulated_path['BAA_AAA_SPREAD_10Y'] + self.simulated_path['AAA_10Y']

            # Fix NaN in the first row (origin) for pct_change columns using real historical data
            origin_idx = start_idx
            prev_idx = origin_idx - 1
            if prev_idx >= 0:
                # Use processed_data for historical values
                prev_gdp_def = self.processed_data.iloc[prev_idx]['GDP_DEF']
                curr_gdp_def = self.processed_data.iloc[origin_idx]['GDP_DEF']
                prev_usd_index = self.processed_data.iloc[prev_idx]['USD_INDEX']
                curr_usd_index = self.processed_data.iloc[origin_idx]['USD_INDEX']
                prev_pce_price = self.processed_data.iloc[prev_idx]['PCE_PRICE_INDEX']
                curr_pce_price = self.processed_data.iloc[origin_idx]['PCE_PRICE_INDEX']
                self.simulated_path.iloc[0, self.simulated_path.columns.get_loc('REAL_GROWTH')] = (curr_gdp_def - prev_gdp_def) / prev_gdp_def
                self.simulated_path.iloc[0, self.simulated_path.columns.get_loc('FX_STRENGTH')] = (curr_usd_index - prev_usd_index) / prev_usd_index
                self.simulated_path.iloc[0, self.simulated_path.columns.get_loc('INFLATION')] = (curr_pce_price - prev_pce_price) / prev_pce_price
        
        # Start from the first forecast step, 0 is reserved for the origin data.
        self.current_index = 1 

        if self.debug:
            print(f"Simulation reset. Start date: {self.simulated_path.index[0]}. Random: {origin is None}")

    def retrieve_origin_data(self):
        """
        Retrieve the macroeconomic data at the origin (start) of the current simulation path.
        
        Returns:
            pd.DataFrame: Data for the first simulated time step.
        """
        if self.simulated_path is None:
            raise RuntimeError("No simulated path available. Call reset() first.")
        return self.simulated_path.iloc[0].to_frame().T

    def retrieve_next_forecast(self, steps=1):
        """
        Retrieve the next forecasted macroeconomic values from the simulation path.

        Args:
            steps (int): Number of steps to retrieve.
        Returns:
            pd.DataFrame: Forecasted values for the next `steps` time periods.
        """
        if self.simulated_path is None:
            raise RuntimeError("No simulated path available. Call reset() first.")
        forecasted_values = self.simulated_path.iloc[self.current_index][self.default_output_columns]
        self.current_index = min(self.current_index + 1, len(self.simulated_path))
        return forecasted_values

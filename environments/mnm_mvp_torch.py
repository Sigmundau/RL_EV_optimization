from __future__ import annotations
import sys
import os
import torch
from tensordict import TensorDict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import datetime
from utils.macro_simulator import Simulation,Dynamic_Simulation
from utils.Company_data_downloader import download_data
from utils.beta_calculator import calculate_beta
from utils.interest_expense_to_debt_predictor import InterestExpensePredictor
import settings
import pandas as pd

from torchrl.envs import EnvBase
from torchrl.data import Unbounded, Categorical, Composite

import warnings


class MnM_MVP_torchrl(EnvBase):
    """
    MnM_MVP is a custom torchrl environment for corporate debt optimization.
    The agent's goal is to manage the firm's debt structure (short-term and long-term debt)
    to maximize the Enterprise Value (EV). The EV is calculated using a modified
    Modigliani-Miller approach with dynamic inputs from a macroeconomic simulator.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self,
                 ticker: str,
                 fred_api_key: str,
                 terminal_growth_rate: float = 0.02,
                 credit_rating: str = 'BAA',
                 simulation_horizon: int = 10,
                 simulation_earliest_origin: datetime.datetime = datetime.datetime(2019, 12, 31),
                 csv_path: str = "data/KO_quarterly.csv",
                 debug: bool = False,
                 device: str = "cpu"):
        """
        Initializes the environment.

        Args:
            ticker (str): The stock ticker of the company to model (e.g., 'AAPL').
            fred_api_key (str): Your API key for the FRED database.
            terminal_growth_rate (g) (float): The rate at which EBIT grows in the terminal phase.
            credit_rating (str): The firm's credit rating ('AAA' or 'BAA').
            simulation_horizon (int): The number of quarters the simulation runs for (episode length).
            simulation_earliest_origin (datetime.datetime): The earliest date possible for the simulation.
            csv_path (str): Path to the CSV file containing company data. If empty, data will be downloaded.
            debug (bool): If True, enables debug mode for additional logging.
            device (str): The device to run the environment on ('cpu' or 'cuda').
        """
        super().__init__(device=device)
        self.debug = debug
        # --- Firm & Model Parameters ---
        self.ticker_str = ticker
        self.g = terminal_growth_rate
        self.credit_rating = credit_rating.upper()
        self.horizon = simulation_horizon
        # self.macro_simulator = Simulation(fred_api_key=fred_api_key, var_order=4, debug=self.debug)
        self.macro_simulator = Dynamic_Simulation(fred_api_key=fred_api_key, var_order=4, debug=self.debug)
        self.simulation_earliest_origin = simulation_earliest_origin
        self.simulation_origin = datetime.datetime.now()
        self.csv_path = csv_path
        self.company_data = None
        self.current_step = 0

        # Company Data
        self.revenue = 0.0
        self.cogs_ratio = 0.0
        self.opex_ratio = 0.0
        self.interest_income = 0.0
        self.interest_income_ratio = 0.0
        self.interest_expense = 0.0
        self.lt_interest_expense_ratio = 0.0
        self.st_interest_expense_ratio = 0.0
        self.depreciation_amortization = 0.0
        self.tax_rate = 0.0
        self.ebit = 0.0
        self.total_asset = 0.0
        self.debt_st = 0.0
        self.debt_lt = 0.0
        self.equity = 0.0
        self.total_asset_lag = 0.0
        self.debt_st_lag = 0.0
        self.debt_lt_lag = 0.0
        self.equity_lag = 0.0
        self.beta_U = 0.0
        self.beta_L = 0.0
        self.capm = []
        self.ev = []

        # Action and Observation Specs
        self._make_spec()

    def _make_spec(self):
        self.observation_spec = Composite(
            {
                "financial_metrics": Unbounded(shape=(11,), dtype=torch.float32),
                "macro_economic_data": Unbounded(shape=(11,), dtype=torch.float32)
            }
        )
        self.action_spec = Composite({
            "action": Categorical(n=3, shape=(2,), dtype=torch.long)
        })
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)


    def _reset(self):
        # Find the simulation origin date
        if self.company_data is None:
            if self.csv_path == "":
                self.company_data = download_data(settings.alpha_vantage_key,
                                                  self.ticker_str)
            else:
                self.company_data = pd.read_csv(self.csv_path, index_col=0)

        datelist = self.company_data['fiscalDateEnding'].to_list()
        datelist_adj = [d for d in datelist if d >= str(self.simulation_earliest_origin)]
        self.simulation_origin = np.random.choice(datelist_adj)
        lag_date = datelist[datelist.index(self.simulation_origin) - 1]
        origin_data = self.company_data.loc[self.company_data['fiscalDateEnding'] == self.simulation_origin]
        lag_data = self.company_data.loc[self.company_data['fiscalDateEnding'] == lag_date]

        # Set the initial values from the balance sheet
        self.total_asset = float(origin_data['totalAssets'].values[0])
        self.debt_st = float(origin_data['shortTermDebt'].values[0])
        self.debt_lt = float(origin_data['longTermDebt'].values[0])
        self.equity = float(origin_data['totalShareholderEquity'].values[0])

        # laggard 1 balance sheet
        self.total_asset_lag = float(lag_data['totalAssets'].values[0])
        self.debt_st_lag = float(lag_data['shortTermDebt'].values[0])
        self.debt_lt_lag = float(lag_data['longTermDebt'].values[0])
        self.equity_lag = float(lag_data['totalShareholderEquity'].values[0])

        # Set the initial values from the income statement
        self.revenue = float(origin_data['totalRevenue'].values[0])
        self.cogs_ratio = float(origin_data['costofGoodsAndServicesSold'].values[0]) / self.revenue
        self.opex_ratio = float(origin_data['operatingExpenses'].values[0]) / self.revenue
        if np.isnan(origin_data['interestIncome'].values[0]):
            self.interest_income = 0.0
        else:
            self.interest_income = float(origin_data['interestIncome'].values[0])
        self.interest_income_ratio = float(self.interest_income / self.total_asset_lag)
        self.interest_expense = float(origin_data['interestExpense'].values[0])
        self.depreciation_amortization = float(origin_data['depreciationAndAmortization'].values[0])
        self.tax_rate = float(origin_data['incomeTaxExpense'].values[0]) / float(origin_data['incomeBeforeTax'].values[0])
        self.ebit = float(origin_data['ebit'].values[0])

        # assume same r in mvp
        self.lt_interest_expense_ratio = self.interest_expense / (self.debt_st + self.debt_lt)
        self.st_interest_expense_ratio = self.lt_interest_expense_ratio


        # market beta calculation
        self.beta_L = calculate_beta(self.ticker_str,
                                    end_date=pd.to_datetime(self.simulation_origin),
                                    years=3)
        self.beta_U = self.beta_L / (1 + (1 - self.tax_rate) * ((self.debt_st + self.debt_lt) / self.equity))

        # step 0 Enterprise Value calculation
        self.nopat = [self.ebit * (1 - self.tax_rate)]

        # Macroeconomic simulator reset
        self.macro_simulator.reset(path_length=self.horizon,
                                   noise_scale=0.5,
                                   origin=pd.to_datetime(self.simulation_origin).date())
        origin = self.macro_simulator.retrieve_origin_data()

        # step 0 CAPM calculation
        self.capm = [origin.EFFR.values[0] + self.beta_U * max(origin.MARKET_RETURN.values[0] - origin.EFFR.values[0], 0)]

        # Reset the current step counter
        self.current_step = 0

        observation = self._get_obs()
      
        return TensorDict({
        "financial_metrics": observation["financial_metrics"],
        "macro_economic_data": observation["macro_economic_data"],
        "done": torch.tensor([False]).view(1, 1),
        "reward": torch.tensor([0.0], dtype=torch.float32).view(1, 1)
    }, batch_size=self.batch_size, device=self.device)

    def _step(self, tensordict: TensorDict):
        self.current_step += 1
        action = tensordict.get("action").cpu().numpy()

        # Store lagged values for balance sheet items
        self.total_asset_lag = self.total_asset
        self.debt_st_lag = self.debt_st
        self.debt_lt_lag = self.debt_lt
        self.equity_lag = self.equity

        # Retrieve macroeconomic data for the current step
        current_macro_data = self.macro_simulator.retrieve_next_forecast(steps=1).iloc[0]

        # Use the real GDP growth rate for revenue model
        real_gdp_growth_rate = current_macro_data['REAL_GROWTH'] / 100  # Convert percentage to decimal

        # Update financial metrics based on models and macro data
        self.revenue = self.revenue_model(real_gdp_growth_rate)
        cogs = self.cogs_model()
        opex = self.opex_model()
        self.interest_income = self.interest_income_model()
        self.interest_expense = self.interest_expense_model()
        depreciation = self.depreciation_model()

        # Calculate EBIT for the current step
        self.ebit = self.revenue - cogs - opex
        pretax_income = self.ebit + self.interest_income - self.interest_expense - depreciation
        nopat = pretax_income * (1 - self.tax_rate)
        # Update the NOPAT for the current step
        self.nopat.append(nopat)

        # Balance Sheet Items Update  
        # Update short-term and long-term debt based on decay models
        st_repayment = self.st_debt_decay_model()
        self.lt_debt_convert_st_model()
        # Assuming only Cash Flow from Operating and Debt repayment from asset
        self.total_asset = self.total_asset + nopat - st_repayment

        # Map discrete actions (0, 1, 2) to percentage point changes (-0.05, 0.00, 0.05)
        action_mapping = {0: -0.05, 1: 0.00, 2: 0.05}
        st_debt_adjustment_ratio = action_mapping[action[0]]
        lt_debt_adjustment_ratio = action_mapping[action[1]]

        # Calculate nominal changes in debt
        delta_debt_st = self.debt_st * st_debt_adjustment_ratio
        delta_debt_lt = self.debt_lt * lt_debt_adjustment_ratio

        # Update debt levels
        self.debt_st += delta_debt_st
        self.debt_lt += delta_debt_lt

        # Ensure debt levels don't go negative
        self.debt_st = max(0.0, self.debt_st)
        self.debt_lt = max(0.0, self.debt_lt)

        self.equity = self.total_asset - (self.debt_st + self.debt_lt)
        bankrupt = False
        if self.equity <= 0:  # Avoid division by zero or negative equity
            bankrupt = True
        else:
            self.beta_U = self.beta_L / (1 + (1 - self.tax_rate) * ((self.debt_st + self.debt_lt) / self.equity))

            # Update CAPM (cost of equity)
            risk_free_rate = current_macro_data['EFFR'] / 100  # Convert percentage to decimal
            market_return = current_macro_data['MARKET_RETURN']
            self.capm.append(risk_free_rate + self.beta_U * max(market_return - risk_free_rate,0))

            cost_of_capital = self.capm[-1]  # Using the latest calculated CAPM
        
        # Check for termination
        terminated = False
        reward = 0.0
        # Only give reward at final step
        if bankrupt:
            terminated = True
            reward = -100.0
        elif self.current_step >= self.horizon:
            terminated = True
            current_ev = 0.0
            if (cost_of_capital - self.g) <= 0:
                terminal_value = 0.0
            else:
                terminal_value = nopat / (cost_of_capital - self.g)
            for i in range(self.horizon):
                current_ev += self.nopat[i] / ((1 + self.capm[i]) ** i)
            current_ev += terminal_value
            reward = current_ev / self.nopat[0]  # Normalize reward by initial NOPAT


        observation = self._get_obs()
        
        out_tensordict = TensorDict(
        source={
            "next": TensorDict({
                "financial_metrics": observation["financial_metrics"],
                "macro_economic_data": observation["macro_economic_data"],
            }, batch_size=self.batch_size, device=self.device),
            "reward": torch.tensor([reward], dtype=torch.float32).view(1, 1),
            "done": torch.tensor([terminated]).view(1, 1),
        },
        batch_size=self.batch_size,
        device=self.device
    )

        return out_tensordict

    def _get_obs(self):
        """
        Helper to get the current observation.
        """
        financial_metrics = torch.tensor([
            self.revenue,
            self.cogs_ratio,
            self.opex_ratio,
            self.interest_income_ratio,
            self.lt_interest_expense_ratio,
            self.st_interest_expense_ratio,
            self.depreciation_amortization,
            self.tax_rate,
            self.total_asset,
            self.debt_st,
            self.debt_lt
        ], dtype=torch.float32)

        # Retrieve the current macroeconomic forecast
        macro_data_df = self.macro_simulator.retrieve_next_forecast(steps=1)
        macro_economic_data = torch.tensor(macro_data_df.iloc[0][self.macro_simulator.default_output_columns].values,
            dtype=torch.float32)

        return {
            "financial_metrics": financial_metrics,
            "macro_economic_data": macro_economic_data
        }

    def _set_seed(self, seed: int):
        """
        Sets the seed for the environment's random number generator.
        """
        rng = torch.manual_seed(seed)
        self.rng = rng

    def revenue_model(self, real_gdp_growth_rate):
        """
        Simple model for revenue growth.
        args:
        real_gdp_growth_rate (float): Real GDP growth rate.

        returns:
        revenue (float): Revenue of the simulation.
        """
        return self.revenue * (1 + real_gdp_growth_rate)

    def cogs_model(self):
        """
        Simple model for cogs, assuming fixed % of revenue.

        returns:
        cogs (float): Cogs of the simulation.
        """
        return self.revenue * self.cogs_ratio

    def opex_model(self):
        """
        Simple model for opex, assuming fixed % of revenue.

        returns:
        opex (float): operating expense of the simulation.
        """
        return self.revenue * self.opex_ratio

    def interest_income_model(self):
        """
        Simple model for interest income, assuming fixed % of total asset in time t-1.

        returns:
        interest_income (float): Interest income of the simulation.
        """
        return self.total_asset_lag * self.interest_income_ratio

    def interest_expense_model(self):
        """
        Simple model for interest expense, assuming fixed % of total LT and ST debt in time t-1.

        returns:
        interest_expense (float): Interest expense of the simulation.
        """
        return self.debt_lt_lag * self.lt_interest_expense_ratio + self.debt_st_lag * self.st_interest_expense_ratio

    def depreciation_model(self):
        """
        Simple model for depreciation, assuming constant from t = 0.

        returns:
        depreciation (float): Depreciation of the simulation.
        """
        return self.depreciation_amortization
    
    def st_debt_decay_model(self, maturity: int = 3):
        """
        Simple model for short-term debt decay, assuming linear decay to zero over a specified maturity period.

        Args:
            maturity (int): The number of periods over which the short-term debt decays to zero.

        Returns:
            repayment (float): The amount of short-term debt that we are required to repay in the current step.
        """
        repayment = self.debt_st / maturity/4
        self.debt_st -= repayment
        return repayment
    
    def lt_debt_convert_st_model(self, maturity: int = 10):
        """
        Simple model for long-term debt decay, assuming linear decay to zero over a specified maturity period.

        Args:
            maturity (int): The number of periods over which the long-term debt decays to zero.

        Returns:
            repayment (float): The amount of long-term debt that we are required to repay in the current step.
        """
        repayment = self.debt_lt / maturity /4
        self.debt_lt -= repayment
        self.debt_st += repayment
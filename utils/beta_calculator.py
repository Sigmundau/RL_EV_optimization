import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def calculate_beta(stock_symbol: str, end_date: pd.Timestamp,market_symbol: str='SPY',years: int = 3) -> float | None:
    """
    Calculates the 3-year daily beta of a stock relative to a market ETF.

    Args:
        stock_symbol (str): The ticker symbol of the stock (e.g., 'KO').
        end_date (pd.Timestamp): The end date for the n-year period as a pandas Timestamp object.
        market_symbol (str): The ticker symbol of the market ETF (default is 'SPY').
        years (int): The number of years to look back for beta calculation (default is 3).
    Returns:
        float | None: The calculated n-year daily beta, or None if data is insufficient.
    """
    # Ensure end_date is a pandas Timestamp; if not, attempt conversion
    if not isinstance(end_date, pd.Timestamp):
        try:
            end_date = pd.to_datetime(end_date)
        except Exception as e:
            print(f"Error: end_date must be a pandas Timestamp or a valid date string. Error: {e}")
            return None

    # Extract date part and format as 'YYYY-MM-DD' string for yfinance
    end_date_str = end_date.strftime('%Y-%m-%d')
    end_dt = end_date.to_pydatetime() # Convert to datetime.datetime object for timedelta calculations

    # Calculate the start date (3 years prior to end_date)
    # Account for leap years by adding 366 days for each leap year within the period.
    # A simpler approximation is 3 * 365 + some padding, or let yfinance handle small discrepancies.
    # For a precise 3 years, subtract 3 years directly from the datetime object.
    start_dt = end_dt - timedelta(days=3 * 365) # Initial approximation
    # Adjust for exact 3 years if necessary, though yfinance handles date ranges flexibly.
    # A more robust way to get exactly 3 years back:
    start_dt = end_dt.replace(year=end_dt.year - 3)
    # If the exact date does not exist (e.g., Feb 29), adjust to Feb 28 or Mar 1
    # yfinance handles the end of month/year adjustments implicitly.

    # Fetch historical data
    stock_data = yf.download(stock_symbol, start=start_dt.strftime('%Y-%m-%d'), end=end_date_str, progress=False)
    market_data = yf.download(market_symbol, start=start_dt.strftime('%Y-%m-%d'), end=end_date_str, progress=False)

    # Check if data was fetched successfully
    if stock_data.empty or market_data.empty:
        print(f"Error: Could not fetch sufficient data for {stock_symbol} or {market_symbol} for the period ending {end_date_str}.")
        return None
    # Calculate daily returns from 'Adj Close' prices
    stock_returns = stock_data['Close'].pct_change().dropna()
    market_returns = market_data['Close'].pct_change().dropna()

    # Combine returns into a single DataFrame for easier calculation
    combined_returns = pd.merge(stock_returns,market_returns,left_index=True, right_index=True, how='inner')

    if combined_returns.empty:
        print("Error: No overlapping daily return data for the specified period after filtering.")
        return None

    # Calculate beta: Covariance(Stock Returns, Market Returns) / Variance(Market Returns)
    covariance_matrix = combined_returns.cov()
    covariance_stock_market = covariance_matrix.loc[stock_symbol, market_symbol]
    variance_market = combined_returns[market_symbol].var()

    if variance_market == 0:
        print(f"Error: Market variance is zero, cannot calculate beta. This might happen with insufficient data or constant market returns over the period.")
        return None

    beta = covariance_stock_market / variance_market

    return beta
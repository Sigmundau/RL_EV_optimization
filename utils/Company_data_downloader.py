import requests
import pandas as pd
def download_data(alpha_vantage_key,symbol):
    """
    Downloads Income and Balance Sheet data from Alpha Vantage and saves it to a CSV file.

    Args:
        alpha_vantage_key (str): Your API key for Alpha Vantage.
        symbol (str): The stock ticker symbol for the company.
    Returns:
        pd.DataFrame: A DataFrame containing the quarterly income statement and balance sheet data. 
    """
    url_income_statement = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol=KO&apikey={alpha_vantage_key}' #
    r_income = requests.get(url_income_statement) 
    data_income = r_income.json() 
    symbol = data_income['symbol'] 

    numeric_cols_income = [
        'grossProfit', 'totalRevenue', 'costOfRevenue', 'costofGoodsAndServicesSold',
        'operatingIncome', 'sellingGeneralAndAdministrative', 'researchAndDevelopment',
        'operatingExpenses', 'investmentIncomeNet', 'netInterestIncome',
        'interestIncome', 'interestExpense', 'nonInterestIncome',
        'otherNonOperatingIncome', 'depreciation', 'depreciationAndAmortization',
        'incomeBeforeTax', 'incomeTaxExpense', 'interestAndDebtExpense',
        'netIncomeFromContinuingOperations', 'comprehensiveIncomeNetOfTax', 'ebit',
        'ebitda', 'netIncome'
    ]

    # Create DataFrame for quarterly income reports
    df_quarterly = pd.DataFrame(data_income['quarterlyReports']) #
    df_quarterly['symbol'] = symbol #
    for col in numeric_cols_income:
        df_quarterly[col] = pd.to_numeric(df_quarterly[col], errors='coerce') #
    df_quarterly['fiscalDateEnding'] = pd.to_datetime(df_quarterly['fiscalDateEnding']) #

    # Convert fiscalDateEnding to datetime objects for consistency (as in your original notebook)
    df_quarterly['fiscalDateEnding'] = df_quarterly['fiscalDateEnding'].apply(lambda x: x.to_pydatetime()) #


    # --- New code for Balance Sheet ---
    url_balance_sheet = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol=KO&apikey={alpha_vantage_key}'
    r_balance = requests.get(url_balance_sheet)
    data_balance = r_balance.json()
    # Define numeric columns for balance sheet (you'll need to inspect the data for exact names)
    numeric_cols_balance = [
        'totalAssets', 'totalCurrentAssets', 
        'cashAndCashEquivalentsAtCarryingValue', 'cashAndShortTermInvestments', 
        'inventory', 'currentNetReceivables', 'totalNonCurrentAssets', 
        'propertyPlantEquipment', 'accumulatedDepreciationAmortizationPPE', 
        'intangibleAssets', 'intangibleAssetsExcludingGoodwill', 'goodwill', 
        'investments', 'longTermInvestments', 'shortTermInvestments', 
        'otherCurrentAssets', 'otherNonCurrentAssets', 
        'totalLiabilities', 'totalCurrentLiabilities', 'currentAccountsPayable', 'deferredRevenue', 
        'currentDebt', 'shortTermDebt', 'totalNonCurrentLiabilities', 'capitalLeaseObligations', 
        'longTermDebt', 'currentLongTermDebt', 'longTermDebtNoncurrent', 'shortLongTermDebtTotal', 
        'otherCurrentLiabilities', 'otherNonCurrentLiabilities', 'totalShareholderEquity', 
        'treasuryStock', 'retainedEarnings', 'commonStock', 'commonStockSharesOutstanding'
    ]

    # Create DataFrame for quarterly balance sheet reports
    df_balance_quarterly = pd.DataFrame(data_balance['quarterlyReports'])
    df_balance_quarterly['symbol'] = symbol
    for col in numeric_cols_balance:
        df_balance_quarterly[col] = pd.to_numeric(df_balance_quarterly[col], errors='coerce')
    df_balance_quarterly['fiscalDateEnding'] = pd.to_datetime(df_balance_quarterly['fiscalDateEnding'])

    # Convert fiscalDateEnding to datetime objects for consistency
    df_balance_quarterly['fiscalDateEnding'] = df_balance_quarterly['fiscalDateEnding'].apply(lambda x: x.to_pydatetime())

    merged_quarterly_df = pd.merge(df_quarterly, df_balance_quarterly,
                                   on=['fiscalDateEnding'],
                                   how='inner', suffixes=('_income', '_balance'))
    return merged_quarterly_df


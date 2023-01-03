"""
Generating Custom Data Bundles via CSV
Logic:
* Get quandl first, up till the second last row
* Get yahoo next, from the last row of quandl

CSV Format: OHLCV
"""

# Libraries
import pandas as pd
import pandas_datareader.data as web
import quandl


# Configs/Constants
tickers = ['AMZN', 'AAPL']
MIN_DATE = pd.to_datetime('1990-01-01')
CSVS_PATH = 'C:\\Users\\tommy.yong\\Documents\\Projects\\python\\ai-tutorials\\algotrading\\quantopian\\data\\daily'


def data_patching(df, ticker):
    """
    Patch missing data from quandl, based on tickers. 
    Potentially to be based on source instead, to do more research.
    """
    QUANDL_MISSING_DATES_SYMBOLS = ['AAPL', 'AMZN']
    if ticker in QUANDL_MISSING_DATES_SYMBOLS:
        df.loc[pd.to_datetime('2017-08-07')] = df.loc[df[df.index < '2017-08-07'].index.max()]
        df.loc[pd.to_datetime('2017-11-08')] = df.loc[df[df.index < '2017-11-08'].index.max()]
    else:
        return df
    df.sort_index(inplace=True)
    return df.loc[MIN_DATE:]
    

def main():
    """
    Main dataflow for fetching input sources and save into CSVs
    """
    for ticker in tickers:
        df_quandl = quandl.get(dataset='WIKI/{}'.format(ticker))
        df_quandl.rename({'Adj. Close':'Adj Close'}, axis=1, inplace=True)
        df_pdr = web.DataReader(ticker, data_source='yahoo', start=df_quandl.index[-1])
        df = pd.concat([
            df_quandl[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']].iloc[:-1], 
            df_pdr[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']][df_quandl.index[-1]:], 
        ])
        df.rename(
            {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume', 'Adj Close': 'adj close'},
            axis=1,
            inplace=True
        )
        df.index.names = ['date']
        df = data_patching(df, ticker)
        df.to_csv('{}/{}.csv'.format(CSVS_PATH, ticker))
    
    
if __name__ == '__main__':
    main()

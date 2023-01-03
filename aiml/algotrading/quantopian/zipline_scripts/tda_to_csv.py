import json
import requests
import pandas as pd


# Configs/Constants
SECRETS_FILE_PATH = 'secrets/secrets.json'
TDA_CLIENT_KEY = 'tda_client_key'
TDA_URL = 'https://api.tdameritrade.com/v1/marketdata/{}/pricehistory'
TICKERS = ['VXX']
CSVS_PATH = 'C:\\Users\\tommy.yong\\Documents\\Projects\\python\\ai-tutorials\\algotrading\\quantopian\\data\\minute'

with open(SECRETS_FILE_PATH) as secrets_file:
	secrets = json.load(secrets_file)


def main():
	for ticker in TICKERS:
		res = requests.get(url=TDA_URL.format(ticker), params={'apikey': secrets[TDA_CLIENT_KEY]})
		df = pd.DataFrame(res.json()['candles'])
		df.rename({'datetime': 'date'}, axis=1, inplace=True)
		df.set_index(pd.to_datetime(df['date'], unit='ms'), inplace=True)
		df.drop('date', inplace=True, axis=1)
		df[['open', 'high', 'low', 'close', 'volume']].to_csv('{}/{}.csv'.format(CSVS_PATH, ticker))


if __name__ == '__main__':
	main()

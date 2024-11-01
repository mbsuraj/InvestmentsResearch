from pycoingecko import CoinGeckoAPI
import pandas as pd
import os
from datetime import datetime, timedelta

cg = CoinGeckoAPI()

CRYPTO_TICKER = "ETH-USD"
PYCOINGECKO_TICKER = "ethereum"
REFRESH_ALL_DATA = False

class CryptoHistory:

    def __init__(self):
        self.refresh_all_data = REFRESH_ALL_DATA
        self.crypto_ticker = CRYPTO_TICKER
        self.pycoingecko_ticker = PYCOINGECKO_TICKER

    def _load_data(self, start_date=None):
        if self.refresh_all_data:
            new = cg.get_coin_market_chart_range_by_id(self.pycoingecko_ticker, vs_currency='usd',
                                                      from_timestamp=1262304000,  # Start date: 2010-01-01
                                                      to_timestamp=int(datetime.now().timestamp()))
        else:
            start_timestamp = datetime.strptime(start_date, "%Y-%m-%d-%H-%M").timestamp()
            new = cg.get_coin_market_chart_range_by_id("ethereum", vs_currency='usd',
                                                      from_timestamp=start_timestamp,
                                                      to_timestamp=int(datetime.now().timestamp()))


        prices = pd.DataFrame(new['prices'], columns=["time", "Price"])
        prices["time"] = pd.to_datetime(prices["time"], unit='ms').dt.date
        prices.columns = ["Date", "Price"]
        return prices.groupby("Date").mean().reset_index()

    def update_data(self):
        if self.refresh_all_data:
            new = self._load_data()
        else:
            old = pd.read_csv(f"{os.getcwd()}/data/{self.crypto_ticker}.csv", usecols=["Date", "Price"])
            start = datetime.strptime(old.iloc[-1, 0], "%Y-%m-%d") + timedelta(days=1)
            if start >= datetime.now():
                raise ValueError("Start cannot be greater than today's date")
            start_date = start.strftime("%Y-%m-%d-%H-%M")
            new = self._load_data(start_date=start_date)
            new = pd.concat([old, new], axis=0)

        new.to_csv(f"{os.getcwd()}/data/{self.crypto_ticker}.csv", index=False)
        return new

# def main():
#     ch = CryptoHistory()
#     ch.update_data()
#
# if __name__ == "__main__":
#     main()

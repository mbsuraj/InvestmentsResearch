from Historic_Crypto import HistoricalData
from datetime import datetime, timedelta
import pandas as pd
import os
pd.DataFrame.append = pd.DataFrame._append

REFRESH_ALL_DATA = False
CRYPTO_TICKER = "ETH-USD"


class CryptoHistory:

    def __init__(self):
        self.refresh_all_data = REFRESH_ALL_DATA
        self.crypto_ticker = CRYPTO_TICKER

    def _load_data(self, start_date=None):
        if self.refresh_all_data:
            new = HistoricalData(self.crypto_ticker, 86400, start_date="2010-01-01",
                                 end_date=datetime.now().strftime("%Y-%m-%d-%H-%M")).retrieve_data()
        else:
            new = HistoricalData('ETH-USD',86400, start_date=start_date,
                                  end_date=datetime.now().strftime("%Y-%m-%d-%H-%M")).retrieve_data()
        new = new.reset_index()
        new = new[["time", "close"]]
        new["time"] = new["time"].dt.date
        new.columns = ["Date", "Price"]

        return new

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

def main():
    ch = CryptoHistory()
    ch.update_data()

if __name__ == "__main__":
    main()
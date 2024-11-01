from indicators import get_momentum, plot_momentum_signals
from robinhoodAPI import CryptoAPITrading, execute_trade_in_dollars
from cryptoHistory import CryptoHistory
import datetime as dt
import util
import pandas as pd


try:
    ch = CryptoHistory()
    ch.update_data()
except ValueError:
    pass


symbol = "ETH-USD"
lookback = 365

sd = dt.datetime(2020, 1, 1)
lookback_timedelta = dt.timedelta(days=lookback)
ed = dt.datetime.now().date()

prices = util.get_data([symbol], dates=pd.date_range(sd - lookback_timedelta, ed))[[symbol]]
prices = prices.fillna(method='bfill')
prices = prices.fillna(method='ffill')

momentum = get_momentum(prices.copy(), sd, lookback=lookback)
plot_momentum_signals(momentum, prices[prices.index >= sd], symbol)
if momentum.iloc[-1, 0] <= 0:
    side = "buy"  # or "sell"
    dollar_amount = 2  # The amount in dollars you want to trade

    # order = execute_trade_in_dollars(symbol, side, dollar_amount)
    # print(order)
else:
    side = "buy"  # or "sell"
    dollar_amount = 1  # The amount in dollars you want to trade

    # order = execute_trade_in_dollars(symbol, side, dollar_amount)
    # print(order)


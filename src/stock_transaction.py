import os
from robinhoodAPI import *

symbol = os.environ["SYMBOL"]
side = os.environ["SIDE"]
trading_client = StockAPITrading()
dollar_amount = int(os.environ["DOLLAR_AMOUNT"])
order = execute_stock_trade_in_dollars(symbol, side, dollar_amount, trading_client=trading_client)
print(order)
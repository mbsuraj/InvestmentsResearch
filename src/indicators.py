import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from src import util
import datetime as dt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def author():
  return 'bmalla7'

def get_bollinger_band_perc(price, sd, lookback=20):
    """
    Calculate the Bollinger Band Percentage (BBP) for a given price series.

    Bollinger Bands are a technical analysis tool used to identify potential overbought or oversold conditions in a financial instrument. BBP is a normalized value that represents the position of the current price within the Bollinger Bands.

    Parameters:
    - price (pandas.Series): A time series of the asset's prices.
    - lookback (int, optional): The number of periods to consider for calculating the moving average and standard deviation. Default is 20.

    Returns:
    - pandas.Series: A series of Bollinger Band Percentage values, indicating the position of each price point relative to the Bollinger Bands. The BBP values range between 0 and 1, where 0 indicates the price is at the lower Bollinger Band, 1 indicates the price is at the upper Bollinger Band, and values in between represent the position within the bands.

    Formula:
    BBP[t] = (price[t] - Lower Bollinger Band[t]) / (Upper Bollinger Band[t] - Lower Bollinger Band[t])
    """
    sma = price.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = price.rolling(window=lookback, min_periods=lookback).std()
    top_band = sma + (2 * rolling_std)
    bottom_band = sma - (2 * rolling_std)
    bbp = (price - bottom_band) / (top_band - bottom_band)
    # res = pd.DataFrame({"bbp": bbp.values.flatten(), "top_band": top_band.values.flatten(), "bottom_band": bottom_band.values.flatten()}, index=price.index)
    return bbp[bbp.index >= sd]

def get_rsi(price, sd, lookback=14):
    """
    Calculate the Relative Strength Index (RSI) for a given price series.

    RSI is a momentum oscillator that measures the speed and change of price movements. It ranges from 0 to 100 and is typically used to identify overbought or oversold conditions in a financial instrument.

    Parameters:
    - price (pandas.Series): A time series of the asset's prices.
    - sd (datetime): to account for any adjustment to price data for lookbacks
    - lookback (int, optional): The number of periods to consider for calculating RSI. Default is 14, which is a common value.

    Returns:
    - pandas.Series: A series of RSI values representing the relative strength of the asset. RSI values range from 0 to 100. Values above 70 often indicate overbought conditions, while values below 30 often indicate oversold conditions.

    Formula:
    - Calculate daily returns: daily_rets[t] = price[t] - price[t-1]
    - Separate gains and losses: gain[t] = max(daily_rets[t], 0) and loss[t] = max(-daily_rets[t], 0)
    - Calculate average gain and average loss over the lookback period.
    - Calculate the Relative Strength (RS) as the ratio of average gain to average loss.
    - Calculate RSI as 100 - (100 / (1 + RS))
    """
    daily_rets = _get_daily_returns(price)
    gain = daily_rets.where(daily_rets > 0, 0)
    loss = -daily_rets.where(daily_rets < 0, 0)
    average_gain = gain.rolling(window=lookback).mean()
    average_loss = loss.rolling(window=lookback).mean()

    rs = average_gain / average_loss
    rsi = 100 - (100 / (1 + rs))
    rsi.iloc[:lookback, :] = np.nan
    rsi[rsi == np.inf] = 100
    return rsi[rsi.index >= sd]

def _get_daily_returns(price):
    """
     Calculate the daily returns from a given price series.

      Daily returns represent the change in the price of an asset from one day to the next, providing insights into the daily price movements.

      Parameters:
      - price (pandas.Series): A time series of the asset's prices.

      Returns:
      - pandas.Series: A series of daily returns, where each value indicates the change in price from the previous day. The first value is set to NaN to represent the lack of a previous day's price for comparison.

    """
    daily_rets = price.copy()
    daily_rets = daily_rets.diff()
    daily_rets.values[0, :] = np.nan
    return daily_rets

def get_macd(prices, sd, short_window=12, long_window=26, signal_window=9):
    """
    Calculate Moving Average Convergence Divergence (MACD) and related indicators for a set of stock prices.

      MACD is a trend-following momentum indicator that helps traders and analysts identify potential trend changes and their strength. It is calculated by subtracting the long-term Exponential Moving Average (EMA) from the short-term EMA. Additionally, the method calculates the Signal Line, which is a smoothed EMA of the MACD, and the MACD Histogram, which is the difference between the MACD and the Signal Line.

      Parameters:
      - prices (pandas.DataFrame): A DataFrame containing the historical prices of multiple stocks, with columns representing different stocks.
      - sd (datetime): to account for any adjustment to price data for lookbacks
      - short_window (int, optional): The short-term EMA window, typically 12 days. Default is 12.
      - long_window (int, optional): The long-t1erm EMA window, typically 26 days. Default is 26.
      - signal_window (int, optional): The window for the Signal Line EMA, typically 9 days. Default is 9.

      Returns:
      - pandas.DataFrame: A DataFrame containing MACD, Signal, and Histogram data for each stock. The columns are labeled with the stock name followed by the indicator name (e.g., 'AAPL_MACD', 'AAPL_Signal', 'AAPL_Histogram').

    """
    # Calculate short-term and long-term EMAs
    short_ema = prices.ewm(span=short_window, adjust=False).mean()
    long_ema = prices.ewm(span=long_window, adjust=False).mean()

    # Calculate MACD line
    macd_line = short_ema - long_ema

    # Calculate Signal line
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()

    # Calculate MACD Histogram
    macd_histogram = macd_line - signal_line

    # Create a DataFrame with MACD, Signal, and Histogram for each stock
    data_frames = []

    for stock in prices.columns:
      macd_data = pd.DataFrame({
        f'{stock}_MACD': macd_line[f'{stock}'],
        f'{stock}_Signal': signal_line[f'{stock}'],
        f'{stock}_Histogram': macd_histogram[f'{stock}']
      })
      data_frames.append(macd_data)

    macd_data = pd.concat(data_frames, axis=1)
    return macd_data[macd_data.index >= sd]


def get_momentum(prices, sd, lookback=14):
    """
    Calculate the momentum of a financial instrument's prices over a specified lookback period.

      Momentum is a measure of the rate of change in the price of an asset over a given historical period. It is often used to identify trends and assess the strength of recent price movements.

      Parameters:
      - prices (pandas.Series or pandas.DataFrame): A time series of the asset's prices, where each row represents a different time point.
      - sd (datetime): to account for any adjustment to price data for lookbacks
      - lookback (int, optional): The number of periods to look back to calculate momentum. Default is 14, which is a common value.

      Returns:
      - pandas.Series or pandas.DataFrame: A series or DataFrame of momentum values, where each value represents the relative change in price over the specified lookback period. If 'prices' is a Series, the returned value will be a Series; if 'prices' is a DataFrame, the returned value will be a DataFrame.

    """
    momentum = (prices / prices.shift(lookback)) - 1
    return momentum[momentum.index >= sd]

def get_stochastic_oscillator_values(prices, sd, k_period=14, d_period=3):
    """
    Calculate the %K and %D values of the Stochastic Oscillator for a set of stock price data.

      The Stochastic Oscillator is a momentum indicator that helps identify overbought and oversold conditions in a financial instrument. It consists of two lines: %K and %D. %K represents the current price's position relative to the high and low prices over a specified period, while %D is a smoothed average of %K.

      Parameters:
      - prices (pandas.DataFrame): A DataFrame containing historical price data, with each row representing different time points and columns for different stocks.
      - sd (datetime): to account for any adjustment to price data for lookbacks
      - k_period (int, optional): The period (typically 14) over which to calculate the Stochastic %K. Default is 14.
      - d_period (int, optional): The period (typically 3) over which to calculate the Stochastic %D. Default is 3.

      Returns:
      - pandas.DataFrame: A DataFrame containing Stochastic %K and %D values for each stock. Columns 'K' and 'D' represent the %K and %D values, respectively.
    """
    # Calculate the highest and lowest prices over the K-period
    prices['Lowest_' + str(k_period)] = prices.min(axis=1).rolling(window=k_period).min()
    prices['Highest_' + str(k_period)] = prices.max(axis=1).rolling(window=k_period).max()

    # Calculate %K
    prices['%K'] = ((prices.iloc[:, 0] - prices['Lowest_' + str(k_period)]) / (prices['Highest_' + str(k_period)] - prices['Lowest_' + str(k_period)]))

    # Calculate %D
    prices['%D'] = prices['%K'].rolling(window=d_period).mean()

    return prices.loc[prices.index >= sd, ['%K', '%D']]

def get_stochastic_oscillator_values_alt(prices, sd, k_period=14, dynamic_d_periods=[3, 7, 10]):
    """
    Calculate the %K and adaptive %D values of the Stochastic Oscillator for a set of stock price data.

    Parameters:
    - prices (pandas.DataFrame): A DataFrame containing historical price data.
    - sd (datetime): Start date for the calculation.
    - k_period (int, optional): The period for %K calculation (typically 14). Default is 14.
    - min_d_period (int, optional): The minimum period for adaptive %D calculation. Default is 3.
    - max_d_period (int, optional): The maximum period for adaptive %D calculation. Default is 14.

    Returns:
    - pandas.DataFrame: A DataFrame containing Stochastic %K and adaptive %D values. Columns 'K' and 'D' represent the %K and adaptive %D values, respectively.
    """

    # Calculate the highest and lowest prices over the K-period
    prices['Lowest_' + str(k_period)] = prices.min(axis=1).rolling(window=k_period).min()
    prices['Highest_' + str(k_period)] = prices.max(axis=1).rolling(window=k_period).max()

    # Calculate %K
    prices['%K'] = ((prices.iloc[:, 0] - prices['Lowest_' + str(k_period)]) / (prices['Highest_' + str(k_period)] - prices['Lowest_' + str(k_period)]))

    # Calculate adaptive %D
    # Calculate dynamic D-period
    rolling_volatility = prices.iloc[:, 0].rolling(
        window=k_period).std()  # Calculate price volatility over the entire range
    # Define conditions and corresponding dynamic D-values
    conditions = [
        rolling_volatility < 0.5,
        (rolling_volatility >= 0.5) & (rolling_volatility < 1.0),
        rolling_volatility >= 1.0
    ]

    # Use np.select to map conditions to dynamic D-values
    dynamic_d_period = np.select(conditions, dynamic_d_periods)


    d_values = []

    for i in range(len(prices)):
        d_period = np.inf if np.isnan(dynamic_d_period[i]) else int(dynamic_d_period[i])
        if (i >= d_period):
            d_value = prices['%K'].iloc[i - d_period:i].mean()
        else:
            d_value = np.nan  # Not enough data for the full dynamic D-period
        d_values.append(d_value)

    prices['%D'] = d_values

    return prices.loc[prices.index >= sd, ['%K', '%D']]

def plot_rsi_indicator(rsi, prices, symbol, lookback):
    date_range = prices.index

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), gridspec_kw={'height_ratios': [3, 1]})
    plt.subplots_adjust(wspace=0, hspace=0)

    # Create a new figure
    # plt.figure(figsize=(20, 6))

    # Plot the RSI values
    ax1.plot(date_range, rsi, label="RSI", color="blue")

    # Plot the normalized prices
    ax2.plot(date_range, prices, label="Price", linestyle='--', color="black")

    # Add horizontal lines at RSI values 70 and 30
    ax1.axhline(70, color='red', linestyle='--')
    ax1.axhline(30, color='forestgreen', linestyle='--')

    # Shade the area above RSI 70 in red
    ax1.fill_between(date_range,
                     70,
                     rsi[symbol],
                     where=(rsi[symbol] >= 70),
                     interpolate=True,
                     color='red',
                     alpha=0.8,
                     label="Overbought (Signal to SELL)"
                     )

    # Shade the area between RSI values 0 and 30 in green
    ax1.fill_between(date_range,
                     30,
                     rsi[symbol],
                     where=(rsi[symbol] <= 30),
                     interpolate=True,
                     color='forestgreen',
                     alpha=0.8,
                     label="Oversold (Signal to BUY)"
                     )

    ax1.fill_between(date_range,
                     30,
                     70,
                     color='grey',
                     alpha=0.3,
                     )

    fig.suptitle(f"{symbol} RSI Indicator Plot: {lookback} day window", fontsize=16)
    # Set zorder for lines in the upper subplot to be below scatter points
    ax1.set_zorder(2)

    ax1.set_xticklabels([])
    ax1.set_ylabel("RSI or Price", fontsize=14)
    ax2.set_xlabel("Date Range", fontsize=14)
    ax2.set_ylabel("Price", fontsize=14)

    ax2.set_ylim(prices[symbol].min() * 0.9, prices[symbol].max() * 1.1)

    date_format = DateFormatter("%Y-%m-%d")  # Specify the date format here
    ax1.xaxis.set_major_formatter(date_format)
    ax2.xaxis.set_major_formatter(date_format)

    ax1.grid(True)
    ax2.grid(True)

    # Show the plot
    # plt.show()

    # Uncomment the next line to save the plot as an image
    plt.savefig(f'../images/{symbol}_rsi_indicator_plot.png')

    # Close the plot
    plt.close()

def plot_stochastic_oscillator_with_prices(k,
                                           d,
                                           prices,
                                           overbought=0.8,
                                           oversold=0.2,
                                           symbol="Stock",
                                           k_period=14,
                                           adaptime_d_period=[3, 14]
                                           ):
    date_range = k.index

    # Create a new figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), gridspec_kw={'height_ratios': [3, 1]})

    plt.subplots_adjust(wspace=0, hspace=0)

    # Plot %K and %D lines in the upper subplot (ax1)
    ax1.plot(date_range, k, label="%K", color='lightblue')
    ax1.plot(date_range, d, label="%D", color='orange')  # Use HEX color code for mustard

    # Add horizontal lines for overbought and oversold levels in the upper subplot (ax1)
    ax1.axhline(overbought, color='red', linestyle='--')
    ax1.axhline(oversold, color='green', linestyle='--')

    # Highlight buy and sell signals in the upper subplot (ax1)
    buy_signal = (k > d) & (k < oversold)
    sell_signal = (k < d) & (k > overbought)
    ax1.scatter(date_range[buy_signal], k[buy_signal], marker='v', color='green', zorder=4, label='Buy Signal', alpha=0.8)
    ax1.scatter(date_range[sell_signal], k[sell_signal], marker='^', color='red', zorder=4, label='Sell Signal', alpha=0.8)

    # buy sell signal regions
    # buy_signal_forshade = (k < oversold) & (k > d)
    # sell_signal_forshade = (k > overbought) & (k < d)

    # # Shade the regions between %K and %D for buy and sell signals
    # fill1 = ax1.fill_between(date_range, k, oversold, where=buy_signal_forshade, color='g', alpha=0.3, label='Buy Signal')
    # fill2 = ax1.fill_between(date_range, k, overbought, where=sell_signal_forshade, color='r', alpha=0.3, label='Sell Signal')

    # Set zorder for lines in the upper subplot to be below scatter points
    ax1.set_zorder(2)

    ax1.set_ylabel("K and D Values", fontsize=14)
    ax2.set_xlabel("Date Range", fontsize=14)
    ax2.set_ylabel("Price", fontsize=14)

    # Plot prices in the lower subplot (ax2)
    ax2.plot(date_range, prices, label="Prices", color='grey')

    # Overlay buy and sell scatter points on the price plot (ax2)
    ax2.scatter(date_range[buy_signal], prices[buy_signal], marker='v', color='green', zorder=4, alpha=0.8)
    ax2.scatter(date_range[sell_signal], prices[sell_signal], marker='^', color='red', zorder=4, alpha=0.8)

    ax2.fill_between(date_range, prices[symbol], 0, where=buy_signal, color='g', alpha=0.8,
                     label='Buy Signal')
    ax2.fill_between(date_range, prices[symbol], 0, where=sell_signal, color='r', alpha=0.8,
                     label='Sell Signal')

    # Adjust y-axis limits for the price subplot (ax2)
    ax2.set_ylim(prices[symbol].min() * 0.9, prices[symbol].max() * 1.1)

    # Increase the size of x-axis labels for better readability in both subplots
    date_format = DateFormatter("%Y-%m-%d")  # Specify the date format here
    ax1.xaxis.set_major_formatter(date_format)
    ax2.xaxis.set_major_formatter(date_format)
    ax1.grid(True)
    ax2.grid(True)

    ax1.set_xticklabels([])

    # Show the legend in the upper subplot (ax1)
    ax1.legend(loc='upper right', bbox_to_anchor=(0.99, 0.98), ncol=4)
    ax2.legend(loc='upper right', bbox_to_anchor=(0.99, 0.98), ncol=4)

    fig.suptitle(f"{symbol} Stochastic Oscillator: K-Period {k_period}, Dynamic D-Period {adaptime_d_period}", fontsize=16)

    # Show the plot
    # plt.show()

    # Uncomment the next line to save the plot as an image
    plt.savefig(f'../images/{symbol}_stochastic_oscillator_with_prices_plot.png')

    # Close the plot
    plt.close()

def plot_macd_signals(macd_df, prices, symbol):
    date_range = macd_df.index

    # Create a new figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), gridspec_kw={'height_ratios': [3, 1]})

    plt.subplots_adjust(wspace=0, hspace=0)

    # Define your buy and sell signals based on your criteria
    buy_signal = macd_df[f'{symbol}_MACD'] >= macd_df[f'{symbol}_Signal']
    sell_signal = macd_df[f'{symbol}_MACD'] <= macd_df[f'{symbol}_Signal']

    # Plot the MACD and Signal lines
    ax1.plot(date_range, macd_df[f'{symbol}_MACD'], label=f'{symbol}_MACD', color='blue')
    ax1.plot(date_range, macd_df[f'{symbol}_Signal'], label=f'{symbol}_Signal', color='orange')
    ax1.bar(date_range, macd_df[f'{symbol}_Histogram'], label=f'{symbol}_Histogram', color='grey')

    ax1.fill_between(macd_df.index, macd_df[f'{symbol}_MACD'], macd_df[f'{symbol}_Signal'], where=buy_signal, color='g', alpha=0.3,
                     label='Buy Signal')
    ax1.fill_between(macd_df.index, macd_df[f'{symbol}_MACD'], macd_df[f'{symbol}_Signal'], where=sell_signal, color='r', alpha=0.3,
                     label='Sell Signal')

    ax1.set_ylabel(f'{symbol} MACD / Signal', fontsize=14)

    ax2.plot(date_range, prices[symbol], label="Price", color='black')
    ax2.fill_between(date_range, prices[symbol], where=buy_signal, color='g', alpha=0.3, label='Buy Signal')
    ax2.fill_between(date_range, prices[symbol], where=sell_signal, color='r', alpha=0.3, label='Sell Signal')

    ax2.set_xlabel("Date", fontsize=14)
    ax2.set_ylabel("Prices", fontsize=14)

    ax2.set_ylim(prices[symbol].min() * 0.9, prices[symbol].max() * 1.1)

    date_format = DateFormatter("%Y-%m-%d")  # Specify the date format here
    ax1.xaxis.set_major_formatter(date_format)
    ax2.xaxis.set_major_formatter(date_format)

    ax1.grid(True)
    ax2.grid(True)

    # remove Date axis for first plot
    ax1.set_xticklabels([])

    fig.suptitle(f'{symbol}_MACD Buy/Sell Signals', fontsize=18)

    # Show the legend in the upper subplot (ax1)
    ax1.legend(loc='upper right', bbox_to_anchor=(0.99, 0.98), ncol=4)
    ax2.legend(loc='upper right', bbox_to_anchor=(0.99, 0.98), ncol=4)

    plt.savefig(f'../images/{symbol}_macd_with_prices_plot.png')
    plt.close()

def plot_momentum_signals(momentum_df, prices_df, symbol, lookback=14):
    date_range = momentum_df.index

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), gridspec_kw={'height_ratios': [3, 1]})
    # plt.subplots_adjust(wspace=0, hspace=0)

    # Define your buy and sell signals based on your momentum criteria
    buy_signal = momentum_df[symbol] >= 0
    sell_signal = momentum_df[symbol] <= 0

    # Plot the momentum indicator
    ax1.plot(date_range, momentum_df[symbol], label='Momentum', color='blue')

    # Use conditional masking to color the momentum line based on buy and sell signals
    ax1.plot(date_range, momentum_df[symbol], label='Momentum', color='blue')
    ax1.fill_between(date_range, momentum_df[symbol], where=buy_signal, color='g', alpha=0.3, label='Buy Signal')
    ax1.fill_between(date_range, momentum_df[symbol], where=sell_signal, color='r', alpha=0.3, label='Sell Signal')

    # Plot the price data
    # plt.plot(datetime, prices_df[symbol], label='Price', color='black')

    ax1.set_ylabel('Momentum', fontsize=14)
    # remove Date axis for first plot
    # ax1.set_xticklabels([])

    ax2.plot(date_range, prices_df[symbol], label='Price', color='black')
    ax2.set_ylim(prices_df[symbol].min() * 0.9, prices_df[symbol].max() * 1.1)

    ax2.fill_between(date_range, prices_df[symbol], where=buy_signal, color='g', alpha=0.3, label='Buy Signal')
    ax2.fill_between(date_range, prices_df[symbol], where=sell_signal, color='r', alpha=0.3, label='Sell Signal')

    ax2.set_xlabel('Date', fontsize=14)
    ax2.set_ylabel('Price', fontsize=14)

    ax1.legend(loc='upper right', bbox_to_anchor=(0.99, 0.98), ncol=4)
    ax2.legend(loc='upper right', bbox_to_anchor=(0.99, 0.98), ncol=3)

    ax1.grid(True)
    ax2.grid(True)

    fig.suptitle(f'{symbol} Momentum Buy/Sell Signals: Lookback {lookback}', fontsize=18)

    plt.savefig(f'{os.getcwd()}/images/{symbol}_momentum_with_prices_plot.png')
    plt.close()

def plot_bbp_signals(bbp_df, prices_df, symbol, lookback=20):
    datetime = bbp_df.index

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), gridspec_kw={'height_ratios': [3, 1]})
    plt.subplots_adjust(wspace=0, hspace=0)

    # Define your buy and sell signals based on your BBP criteria
    buy_signal = bbp_df[symbol] <= 0
    sell_signal = bbp_df[symbol] >= 1

    # Plot the BBP
    ax1.plot(datetime, bbp_df, label='BBP', color='grey')
    # Add horizontal lines at 1 and 0
    ax1.axhline(1, color='red', linestyle='--')
    ax1.axhline(0, color='green', linestyle='--')

    # Use conditional masking to color the BBP line based on buy and sell signals
    ax1.fill_between(datetime, bbp_df[symbol], 0, where=buy_signal, color='forestgreen', alpha=0.8, label='Buy Signal')
    ax1.fill_between(datetime, bbp_df[symbol], 1, where=sell_signal, color='darkred', alpha=0.8, label='Sell Signal')
    # ax1.set_xlabel('Date')
    # remove Date axis for first plot
    ax1.set_xticklabels([])
    ax1.set_ylabel('BBP', fontsize=14)

    ax2.plot(datetime, prices_df[symbol], label="Price", color="black")
    ax2.fill_between(datetime, prices_df[symbol], 0, where=buy_signal, color='forestgreen', alpha=0.8, label='Buy Signal')
    ax2.fill_between(datetime, prices_df[symbol], 0, where=sell_signal, color='darkred', alpha=0.8, label='Sell Signal')

    ax2.set_xlabel('Date', fontsize=14)
    ax2.set_ylabel('Price', fontsize=14)

    ax1.legend(loc='upper right', bbox_to_anchor=(0.99, 0.98), ncol=4)
    ax2.legend(loc='upper right', bbox_to_anchor=(0.99, 0.98), ncol=3)
    plt.suptitle(f'{symbol} BBP Buy/Sell Signals: Lookback {lookback}', fontsize=18)

    ax1.grid()
    ax2.grid()

    plt.savefig(f'../images/{symbol}_bbp_with_prices_plot.png')
    plt.close()

def run():
    """
    run all indicators and generate their respective charts.

    """
    symbol = "ETH-USD"
    sd = dt.datetime(2020, 1, 1)
    lookback_timedelta = dt.timedelta(days=30)
    ed = dt.datetime(2024, 9, 1)

    prices = util.get_data([symbol], dates=pd.date_range(sd - lookback_timedelta, ed))[[symbol]]
    prices = prices.fillna(method='bfill')
    prices = prices.fillna(method='ffill')
    rsi = get_rsi(prices.copy(), sd, lookback=14)
    plot_rsi_indicator(rsi, prices[prices.index >= sd], symbol, 14)

    sov = get_stochastic_oscillator_values_alt(prices.copy(),
                                             sd,
                                             k_period=14,
                                             dynamic_d_periods=[5, 7, 10])
    plot_stochastic_oscillator_with_prices(sov["%K"],
                                         sov["%D"],
                                         prices[prices.index >= sd],
                                         overbought=0.8,
                                         oversold=0.2,
                                         symbol=symbol,
                                         k_period=14,
                                         adaptime_d_period=[5, 7, 10]
                                         )

    # normalize the price data
    # prices = prices / prices.iloc[0, :]

    macd = get_macd(prices.copy(), sd, long_window=26, short_window=12, signal_window=9)
    plot_macd_signals(macd, prices[prices.index >= sd], symbol)
    momentum = get_momentum(prices.copy(), sd, lookback=365)
    plot_momentum_signals(momentum, prices[prices.index >= sd], symbol)
    bbp = get_bollinger_band_perc(prices.copy(), sd, lookback=20)
    plot_bbp_signals(bbp, prices[prices.index >= sd], symbol, lookback=20)

if __name__ == "__main__":
    run()
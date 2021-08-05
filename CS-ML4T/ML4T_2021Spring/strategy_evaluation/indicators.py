"""
Student Name: Aarushi Gupta (replace with your name)
GT User ID: agupta857 (replace with your User ID)
GT ID: 903633934 (replace with your GT ID)
"""
import numpy as np
import pandas as pd
from util import get_data, plot_data
import matplotlib.pyplot as plt


def author():
    return 'agupta857'


def moving_standard_deviation(prices, lookback=15):
    return prices.rolling(window=lookback).std()


def simple_mov_avg(prices, lookback=15):
    return prices.rolling(window=lookback).mean()


def price_per_sma(prices, lookback=15):
    daily_sma = simple_mov_avg(prices, lookback)
    ppsma = prices / daily_sma
    return ppsma


def bollinger_band_value(prices, lookback=15):
    sma = simple_mov_avg(prices, lookback)
    std = moving_standard_deviation(prices, lookback)
    return (prices - sma) / (2 * std)


def exp_mov_avg(prices, lookback=15):
    initial_sma = prices.iloc[:lookback, 0].cumsum()
    divisor = np.arange(lookback)+1
    initial_sma = initial_sma.values / divisor
    multiplier = 2 / (lookback + 1)

    ema = prices.copy()
    ema.iloc[:lookback, 0] = initial_sma
    for i in range(lookback, prices.shape[0]):
        cost = ema.iloc[i, 0]
        prev_ema = ema.iloc[i-1, 0]
        ema.iloc[i, 0] = (cost - prev_ema)*multiplier + prev_ema

    return ema


def ma_conv_div(prices):
    ema_12day = exp_mov_avg(prices, 12)
    ema_26day = exp_mov_avg(prices, 26)

    return ema_12day - ema_26day


def momentum(prices, lookback=15):
    mom = prices / prices.shift(lookback)
    return mom


def rel_strength_idx(prices, lookback=15):
    daily_returns = prices - prices.shift(1).fillna(0)
    cum_gain = daily_returns[daily_returns >= 0].fillna(0).cumsum()
    cum_loss = -1 * daily_returns[daily_returns < 0].fillna(0).cumsum()

    lookback_gain = cum_gain - cum_gain.shift(lookback).fillna(0)
    lookback_gain.values[:lookback, :] = 0
    lookback_loss = cum_loss - cum_loss.shift(lookback).fillna(0)
    lookback_loss.values[:lookback, :] = 0

    avg_gain = lookback_gain / lookback
    avg_loss = lookback_loss / lookback

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def getCombinedIndicators(symbols, sd, ed, lookback=15):
    dates = pd.date_range(sd, ed)
    prices = get_data(symbols, dates)
    prices = pd.DataFrame(prices[symbols])
    normed_prices = prices / prices.iloc[0]

    sma = simple_mov_avg(normed_prices, lookback)
    std = moving_standard_deviation(normed_prices, lookback)

    # MACD Indicator
    macd = ma_conv_div(normed_prices)

    # Bollinger Bands Indicator
    bbvalue = bollinger_band_value(normed_prices, lookback)

    # RSI Indicator
    rsi = rel_strength_idx(normed_prices, lookback)

    RSI_LOW = 30
    RSI_HIGH = 60

    BBV_LOW = -0.7
    BBV_HIGH = 0.7

    signal_rsi = rsi.copy()
    signal_rsi[:] = 0
    signal_rsi[rsi <= RSI_LOW] = 1
    signal_rsi[rsi >= RSI_HIGH] = -1

    signal_bbv = bbvalue.copy()
    signal_bbv[:] = 0
    signal_bbv[bbvalue <= BBV_LOW] = 1
    signal_bbv[bbvalue >= BBV_HIGH] = -1

    signal = exp_mov_avg(macd, 9)
    diff = np.diff(np.sign(macd - signal), axis=0)
    oversold = np.where(diff > 0)[0]
    overbought = np.where(diff < 0)[0]

    c_overbought = []
    c_oversold = []
    for i in overbought:
        if signal_rsi.iloc[i, 0] == -1 or signal_bbv.iloc[i, 0] == -1:
            c_overbought.append(i)
    for i in oversold:
        if signal_rsi.iloc[i, 0] == 1 or signal_bbv.iloc[i, 0] == 1:
            c_oversold.append(i)

    cross_dates_oversold = prices.index[c_oversold]
    cross_dates_overbought = prices.index[c_overbought]

    print("sell: ", cross_dates_overbought)
    print("buy: ", cross_dates_oversold)
    cross_dates_short = cross_dates_overbought

    plt.figure(6, figsize=(20, 20))
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(20,15))

    ax1.plot(normed_prices, label="JPM price", color='gray')
    ax1.plot(sma, label="Simple Moving Avg", color='green')
    ax1.plot(sma + 2 * std, color='orange', ls='--')
    ax1.plot(sma - 2 * std, color='orange', ls='--')
    ax1.plot([], [], label="Bollinger Bands", color='orange')
    ax1.vlines(cross_dates_oversold, ymin=0.6, ymax=1.2, color='green', linestyle='dotted')
    ax1.vlines(cross_dates_overbought, ymin=0.6, ymax=1.2, color='red', linestyle='dotted')
    ax1.vlines(cross_dates_short, ymin=0.6, ymax=1.2, color='black', linestyle='dotted')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.legend()

    ax2.plot(rsi, label="RSI", color="orange")
    ax2.axhline(y=RSI_HIGH, label="Overbought", color='red', linestyle='--')
    ax2.axhline(y=RSI_LOW, label="Underbought", color='red', linestyle='dotted')
    ax2.vlines(cross_dates_oversold, ymin=0, ymax=100, color='green', linestyle='dotted')
    ax2.vlines(cross_dates_overbought, ymin=0, ymax=100, color='red', linestyle='dotted')
    ax2.vlines(cross_dates_short, ymin=0, ymax=100, color='black', linestyle='dotted')
    ax2.set_xlabel("Date")
    ax2.set_ylabel("RSI")
    ax2.legend()

    ax3.plot(macd, label="MACD", color="blue")
    ax3.plot(exp_mov_avg(macd, 9), label="EMA 9 MACD", color="black")
    ax3.vlines(cross_dates_oversold, ymin=-0.1, ymax=0.1, color='green', linestyle='dotted')
    ax3.vlines(cross_dates_overbought, ymin=-0.1, ymax=0.1, color='red', linestyle='dotted')
    ax3.vlines(cross_dates_short, ymin=-0.1, ymax=0.1, color='black', linestyle='dotted')
    ax3.axhline(y=0, label="Overbought", color='red', linestyle='--')
    ax3.legend()

    ax4.plot(bbvalue, label="BB value (price-mean/2*std)", color="orange")
    ax4.axhline(y=[BBV_HIGH], color='gray', linestyle='--')
    ax4.axhline(y=[BBV_LOW], color='gray', linestyle='--')
    ax4.vlines(cross_dates_oversold, ymin=-1.5, ymax=1.5, color='green', linestyle='dotted')
    ax4.vlines(cross_dates_overbought, ymin=-1.5, ymax=1.5, color='red', linestyle='dotted')
    ax4.vlines(cross_dates_short, ymin=-1.5, ymax=1.5, color='black', linestyle='dotted')
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Normalized Price")
    ax4.legend()

    plt.title('JPM')

    plt.savefig("COMBINATION.png")


if __name__ == "__main__":
    symbols = ["JPM"]
    start_date = "2008-1-1"
    end_date = "2009-12-31"

    getCombinedIndicators(symbols, start_date, end_date, lookback=20)

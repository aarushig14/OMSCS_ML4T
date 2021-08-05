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

    return ema_26day - ema_12day


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


def getIndicators(symbols, sd, ed, lookback=15):
    dates = pd.date_range(sd, ed)
    prices = get_data(symbols, dates, False)
    prices = prices.fillna(method='ffill').fillna(method='bfill')
    normed_prices = prices / prices.iloc[0]

    sma = simple_mov_avg(normed_prices, lookback)
    std = moving_standard_deviation(normed_prices, lookback)

    # Exponential Moving Average
    ema = exp_mov_avg(normed_prices, lookback)

    plt.figure(1, figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(normed_prices, label="JPM price", color='gray')
    plt.plot(ema, label="Exponential Moving Average", color="green")
    plt.title("JPM")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(ema, label="Exponential Moving Average", color="green")
    plt.plot(sma, label="Simple Moving Avg", color='orange', linestyle='--')
    plt.axhline(y=[1.0], color='gray', linestyle='--')
    plt.title("JPM")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.savefig("EMA.png")

    # MACD Indicator
    macd = ma_conv_div(normed_prices)
    ema_12 = exp_mov_avg(normed_prices, 12)
    ema_26 = exp_mov_avg(normed_prices, 26)

    plt.figure(4, figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(normed_prices, label="JPM price", color='gray')
    plt.plot(ema_12, label="12 Period EMA", color='green')
    plt.plot(ema_26, label="26 Period EMA", color="orange")
    plt.title("JPM")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(macd, label="MACD", color="orange")
    plt.axhline(y=[0.0], color='gray', linestyle='--')
    plt.title("JPM")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.savefig("MACD.png")

    # Bollinger Bands Indicator
    bbvalue = bollinger_band_value(normed_prices, lookback)

    plt.figure(2, figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(normed_prices, label="JPM price", color='gray')
    plt.plot(sma, label="Simple Moving Avg", color='green')
    plt.plot(sma + 2 * std, color='orange', ls='--')
    plt.plot(sma - 2 * std, color='orange', ls='--')
    plt.plot([], [], label="Bollinger Bands", color='orange')
    plt.title("JPM")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(bbvalue, label="BB value (price-mean/2*std)", color="orange")
    plt.axhline(y=[1.0], color='gray', linestyle='--')
    plt.axhline(y=[-1.0], color='gray', linestyle='--')
    plt.title("JPM")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.savefig("BB.png")

    # Momentum Indicator
    mom = momentum(normed_prices, lookback)

    plt.figure(3, figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(normed_prices, label="JPM price", color='gray')
    plt.plot(sma, label="Simple Moving Avg", color='green')
    plt.plot(mom, label="Momentum", color="orange")
    plt.title("JPM")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(mom, label="Momentum", color="orange")
    plt.plot(sma, label="Simple Moving Avg", color='green', linestyle='dotted')
    plt.axhline(y=[1.0], color='gray', linestyle='--')
    plt.title("JPM")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.savefig("Momentum.png")

    # RSI Indicator
    rsi = rel_strength_idx(normed_prices, lookback)

    plt.figure(5, figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(normed_prices, label="JPM price", color='gray')
    plt.plot(sma, label="Simple Moving Avg", color='green')
    # plt.plot(rsi, label="RSI", color="orange")
    plt.title("JPM")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(rsi, label="RSI", color="orange")
    plt.axhline(y=70, label="Overbought", color='red', linestyle='--')
    plt.axhline(y=30, label="Underbought", color='red', linestyle='dotted')
    plt.title("JPM")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.savefig("RSI.png")


if __name__ == "__main__":
    symbols = ["JPM"]
    start_date = "2008-1-1"
    end_date = "2009-12-31"

    getIndicators(symbols, start_date, end_date, lookback=20)

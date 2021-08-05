"""
Student Name: Aarushi Gupta (replace with your name)
GT User ID: agupta857 (replace with your User ID)
GT ID: 903633934 (replace with your GT ID)
"""

import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from indicators import rel_strength_idx, ma_conv_div, exp_mov_avg, bollinger_band_value
from marketsimcode import compute_portvals, assess_portfolio
from util import get_data


def get_benchmark(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
    dates = pd.date_range(sd, ed)
    prices = get_data([symbol], dates)
    prices = pd.DataFrame(prices[symbol])
    orders = [1000, -1000]
    date = [prices.index[0], prices.index[-1]]

    orders = pd.DataFrame(data=orders, index=date)
    benchmark = compute_portvals(symbol=symbol, orders=orders, start_val=sv, commission=9.95, impact=0.005)
    return benchmark


def testPolicy(symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
    dates = pd.date_range(sd, ed)
    prices = get_data([symbol], dates)
    prices = pd.DataFrame(prices[symbol])
    normed_prices = prices / prices.iloc[0]
    orders = []
    date = []
    total = 0

    rsi = rel_strength_idx(normed_prices, lookback=20)
    bbvalue = bollinger_band_value(normed_prices, lookback=20)
    macd = ma_conv_div(normed_prices)

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
    crossovers = np.argwhere(diff)[:, 0]
    signal_macd = macd.copy()
    signal_macd.iloc[:] = 0
    for i in crossovers:
        if diff[i] > 0:
            signal_macd.iloc[i, 0] = 1
        elif diff[i] < 0:
            signal_macd.iloc[i, 0] = -1

    for i in range(prices.shape[0] - 1):
        buy = signal_macd.iloc[i, 0] == 1 and (signal_rsi.iloc[i, 0] == 1 or signal_bbv.iloc[i, 0] == 1)
        sell = signal_macd.iloc[i, 0] == -1 and (signal_rsi.iloc[i, 0] == -1 or signal_bbv.iloc[i, 0] == -1)

        if total == 0:
            if buy:
                orders.append(1000)
                date.append(prices.index[i])
                total = total + 1000
            elif sell:
                orders.append(-1000)
                date.append(prices.index[i])
                total = total - 1000
        elif total == 1000:
            if sell:
                orders.append(-2000)
                date.append(prices.index[i])
                total = total - 2000
        elif total == -1000:
            if buy:
                orders.append(2000)
                date.append(prices.index[i])
                total = total + 2000

    if total == 1000:
        orders.append(-1000)
        date.append(prices.index[-1])
    elif total == -1000:
        orders.append(1000)
        date.append(prices.index[-1])

    orders = pd.DataFrame(data=orders, index=date)
    return orders


def testStrategy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000, verbose=False, name='Portvals_MS.png'):
    orders = testPolicy(symbol, sd, ed, sv)
    portvals = compute_portvals(symbol=symbol, orders=orders, start_val=sv, commission=9.95, impact=0.005)
    benchmark = get_benchmark(symbol, sd, ed, sv)

    portvals = portvals / portvals.iloc[0]
    benchmark = benchmark / benchmark.iloc[0]

    ymin = min(np.min(benchmark), np.min(portvals))
    ymax = max(np.max(benchmark), np.max(portvals))

    buy = orders[orders[0] > 0].index
    sell = orders[orders[0] < 0].index

    plt.figure(figsize=(8, 6))
    plt.plot(portvals, label='Portfolio', color='red')
    plt.plot(benchmark, label='Benchmark', color='green')
    plt.vlines(buy, ymin=ymin, ymax=ymax, color='blue', linestyle='dotted', label='LONG')
    plt.vlines(sell, ymin=ymin, ymax=ymax, color='black', linestyle='dotted', label='SHORT')
    plt.title('JPM')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.savefig(name)

    port_cr, port_adr, port_std, port_sr = assess_portfolio(portvals)
    bench_cr, bench_adr, bench_std, bench_sr = assess_portfolio(benchmark)

    if verbose:
        print('orders -\n', orders)
        # Compare portfolio against Benchmark
        print(f"Date Range: {sd} to {ed}")
        print()
        print(f"Sharpe Ratio of Fund: {port_sr}")
        print(f"Sharpe Ratio of Benchmark : {bench_sr}")
        print()
        print(f"Cumulative Return of Fund: {port_cr}")
        print(f"Cumulative Return of Benchmark : {bench_sr}")
        print()
        print(f"Standard Deviation of Fund: {port_std}")
        print(f"Standard Deviation of Benchmark : {bench_std}")
        print()
        print(f"Average Daily Return of Fund: {port_adr}")
        print(f"Average Daily Return of Benchmark : {bench_adr}")
        print()
        print(f"Start Value: {sv}")
        print(f"Final Portfolio Value: {portvals.iloc[-1] * sv}")
        print(f"Final Benchmark Value: {benchmark.iloc[-1] * sv}")


def author():
    return 'agupta857'


if __name__ == "__main__":
    symbol = "JPM"
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    start_val = 100000

    testStrategy(symbol, start_date, end_date, start_val, name='Portvals_MS_Insample.png',verbose=True)
    testStrategy(symbol, dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31), start_val, name='Portvals_MS_Outsample.png', verbose=True)

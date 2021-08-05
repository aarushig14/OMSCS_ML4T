"""
Student Name: Aarushi Gupta (replace with your name)
GT User ID: agupta857 (replace with your User ID)
GT ID: 903633934 (replace with your GT ID)
"""

import datetime as dt
import pandas as pd
import numpy as np
from util import get_data, plot_data
import matplotlib.pyplot as plt
from marketsimcode import compute_portvals, assess_portfolio
from indicators import rel_strength_idx, ma_conv_div, exp_mov_avg, bollinger_band_value, simple_mov_avg, price_per_sma


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
    RSI_HIGH = 80

    BBV_LOW = -0.9
    BBV_HIGH = 0.8

    signal_rsi = rsi.copy()
    signal_rsi[:] = 0
    signal_rsi[rsi <= RSI_LOW] = 1
    signal_rsi[rsi >= RSI_HIGH] = -1

    signal_bbv = bbvalue.copy()
    signal_bbv[:] = 0
    signal_bbv[bbvalue <= BBV_LOW] = 1
    signal_bbv[bbvalue >= BBV_HIGH] = -1

    signal_macd = macd.copy()
    signal = exp_mov_avg(macd, 9)
    for i in range(macd.shape[0]):
        if 0 <= signal.iloc[i, 0] <= macd.iloc[i, 0] or macd.iloc[i, 0] <= signal.iloc[i, 0] <= 0:
            signal_macd.iloc[i, 0] = -1
        elif signal.iloc[i, 0] >= macd.iloc[i, 0] >= 0:
            signal_macd.iloc[i, 0] = 1
        else:
            signal_macd.iloc[i, 0] = 0

    for i in range(prices.shape[0] - 1):
        buy = signal_rsi.iloc[i, 0] == 1 or signal_bbv.iloc[i, 0] == 1
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
    print(orders)
    portvals = compute_portvals(symbol=symbol, orders=orders, start_val=sv, commission=9.95, impact=0.005)
    return portvals


def testStrategy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
    portvals = testPolicy(symbol, start_date, end_date, start_val)
    benchmark = get_benchmark(symbol, start_date, end_date, start_val)

    portvals = portvals / portvals.iloc[0]
    benchmark = benchmark / benchmark.iloc[0]

    plt.figure(figsize=(8, 6))
    plt.plot(portvals, label='Portfolio', color='red')
    plt.plot(benchmark, label='Benchmark', color='green')
    plt.title('JPM')
    plt.xlabel('Date')
    plt.ylabel('Nomralized Price')
    plt.legend()
    plt.savefig('Portvals.png')

    port_cr, port_adr, port_std, port_sr = assess_portfolio(portvals)
    bench_cr, bench_adr, bench_std, bench_sr = assess_portfolio(benchmark)

    # Compare portfolio against Benchmark
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {port_cr}")
    print(f"Sharpe Ratio of Benchmark : {bench_cr}")
    print()
    print(f"Cumulative Return of Fund: {port_adr}")
    print(f"Cumulative Return of Benchmark : {bench_adr}")
    print()
    print(f"Standard Deviation of Fund: {port_std}")
    print(f"Standard Deviation of Benchmark : {bench_std}")
    print()
    print(f"Average Daily Return of Fund: {port_sr}")
    print(f"Average Daily Return of Benchmark : {bench_sr}")
    print()
    print(f"Start Value: {start_val}")
    print(f"Final Portfolio Value: {portvals.iloc[-1] * start_val}")
    print(f"Final Benchmark Value: {benchmark.iloc[-1] * start_val}")


def author():
    return 'agupta857'


if __name__ == "__main__":
    symbol = "JPM"
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    start_val = 100000

    testStrategy(symbol, start_date, end_date, start_val)

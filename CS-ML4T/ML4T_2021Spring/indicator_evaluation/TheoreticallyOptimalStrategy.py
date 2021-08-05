"""
Student Name: Aarushi Gupta (replace with your name)
GT User ID: agupta857 (replace with your User ID)
GT ID: 903633934 (replace with your GT ID)
"""

import datetime as dt
import pandas as pd
from util import get_data, plot_data
import matplotlib.pyplot as plt
from marketsimcode import compute_portvals, assess_portfolio


def get_portfolio(orders, sv):
    portvals = compute_portvals(orders, sv, 0.0, 0.0)
    return portvals


def get_benchmark(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
    prices = get_data([symbol], pd.date_range(sd, ed))
    prices = prices[symbol]
    orders = {'Symbol': [symbol, symbol],
              'Order': ['BUY', 'SELL'],
              'Shares': [1000, 1000]}
    date = [prices.index[0], prices.index[-1]]

    orders = pd.DataFrame(data=orders, index=date)
    benchmark = compute_portvals(orders, sv, 0.0, 0.0)
    return benchmark


def testPolicy(symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
    dates = pd.date_range(sd, ed)
    prices = get_data([symbol], dates)
    prices = prices[symbol]
    orders = {'Symbol': [],
              'Order': [],
              'Shares': []}
    date = []
    total = 0

    for i in range(prices.shape[0] - 1):
        curr = prices.iloc[i]
        next_day = prices.iloc[i + 1]
        if curr < next_day:
            if total == -1000:
                orders['Symbol'].append(symbol)
                orders['Order'].append('BUY')
                orders['Shares'].append(2000)
                date.append(prices.index[i])
                total = total + 2000
            elif total == 0:
                orders['Symbol'].append(symbol)
                orders['Order'].append('BUY')
                orders['Shares'].append(1000)
                date.append(prices.index[i])
                total = total + 1000
        elif curr > next_day:
            if total == 1000:
                orders['Symbol'].append(symbol)
                orders['Order'].append('SELL')
                orders['Shares'].append(2000)
                date.append(prices.index[i])
                total = total - 2000
            elif total == 0:
                orders['Symbol'].append(symbol)
                orders['Order'].append('SELL')
                orders['Shares'].append(1000)
                date.append(prices.index[i])
                total = total - 1000

    if total == 1000:
        orders['Symbol'].append(symbol)
        orders['Order'].append('SELL')
        orders['Shares'].append(1000)
        date.append(prices.index[i])
        total = total - 1000
    elif total == -1000:
        orders['Symbol'].append(symbol)
        orders['Order'].append('BUY')
        orders['Shares'].append(1000)
        date.append(prices.index[i])
        total = total + 1000

    orders = pd.DataFrame(data=orders, index=date)
    portvals = compute_portvals(orders, sv, 0.0, 0.0)
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

"""
Student Name: Aarushi Gupta (replace with your name)
GT User ID: agupta857 (replace with your User ID)
GT ID: 903633934 (replace with your GT ID)
"""
import numpy as np
import pandas as pd
from util import get_data, plot_data
import matplotlib.pyplot as plt
import datetime as dt
from indicators import getIndicators
from marketsimcode import assess_portfolio
import TheoreticallyOptimalStrategy as tos


def author():
    return 'agupta857'


def test_code(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000, lookback=20):

    """ INDICATORS.PY """
    getIndicators([symbol], sd, ed, lookback)

    """ THEORETICALLYOPTIMALSTRATEGY.PY """
    portvals = tos.testPolicy(symbol, sd, ed, sv)
    benchmark = tos.get_benchmark(symbol, sd, ed, sv)

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
    print(f"Date Range: {sd} to {ed}")
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
    print(f"Start Value: {sv}")
    print(f"Final Portfolio Value: {portvals.iloc[-1]}")
    print(f"Final Benchmark Value: {benchmark.iloc[-1]}")


if __name__ == '__main__':
    symbols = ["JPM"]
    start_date = "2008-1-1"
    end_date = "2009-12-31"
    start_val = 100000
    lookback=30

    test_code(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000, lookback=20)

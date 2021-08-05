"""
Student Name: Aarushi Gupta (replace with your name)
GT User ID: agupta857 (replace with your User ID)
GT ID: 903633934 (replace with your GT ID)
"""

import datetime as dt
import matplotlib.pyplot as plt
import random

from ManualStrategy import get_benchmark, testPolicy
import StrategyLearner as sl
from marketsimcode import compute_portvals, assess_portfolio


def compare_strategies(verbose=False):
    symbol = "JPM"
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    start_val = 100000
    impact = 0.005
    commission = 9.95

    # Benchmark
    benchmark = get_benchmark(symbol, start_date, end_date, start_val)

    # Manual Strategy
    ms_trades = testPolicy(symbol, start_date, end_date, start_val)
    portvals_MS = compute_portvals(symbol, ms_trades, start_val, commission=commission, impact=impact)

    # Strategy Learner
    learner = sl.StrategyLearner(verbose=False, commission=commission, impact=impact)
    learner.add_evidence(symbol, start_date, end_date, start_val)
    sl_trades = learner.testPolicy(symbol, start_date, end_date, start_val)
    portvals_SL = compute_portvals(symbol, sl_trades, start_val, commission=commission, impact=impact)

    # Normalize
    benchmark = benchmark / benchmark.iloc[0]
    portvals_MS = portvals_MS / portvals_MS.iloc[0]
    portvals_SL = portvals_SL / portvals_SL.iloc[0]

    plt.figure(figsize=(8, 6))
    plt.plot(benchmark, label='Benchmark', color='orange')
    plt.plot(portvals_MS, label='Manual Strategy', color='black')
    plt.plot(portvals_SL, label='Strategy Learner (RT + BagLearner)', color='green')
    plt.title('JPM')
    plt.xlabel('Date')
    plt.ylabel('Nomralized Price')
    plt.legend()
    plt.savefig('experiment1.png')

    ms_cr, ms_adr, ms_std, ms_sr = assess_portfolio(portvals_MS)
    sl_cr, sl_adr, sl_std, sl_sr = assess_portfolio(portvals_SL)
    bench_cr, bench_adr, bench_std, bench_sr = assess_portfolio(benchmark)

    if verbose:
        print(' Results saved in experiment1.png ')
        # Compare portfolio against Benchmark
        print(f"Date Range: {start_date} to {end_date}")
        print()
        print(f"Sharpe Ratio of Fund - Manual Strategy: {ms_sr}")
        print(f"Sharpe Ratio of Fund - Strategy learner: {sl_sr}")
        print(f"Sharpe Ratio of Fund - Benchmark : {bench_sr}")
        print()
        print(f"Cumulative Return of Fund - Manual Strategy: {ms_cr}")
        print(f"Cumulative Return of Fund - Strategy learner: {sl_cr}")
        print(f"Cumulative Return of Fund - Benchmark : {bench_cr}")
        print()
        print(f"Standard Deviation of Fund - Manual Strategy: {ms_std}")
        print(f"Standard Deviation of Fund - Strategy learner: {sl_std}")
        print(f"Standard Deviation of Fund - Benchmark : {bench_std}")
        print()
        print(f"Average Daily Return of Fund - Manual Strategy: {ms_adr}")
        print(f"Average Daily Return of Fund - Strategy learner: {sl_adr}")
        print(f"Average Daily Return of Fund - Benchmark : {bench_adr}")
        print()
        print(f"Start Value: {start_val}")
        print(f"Final Portfolio Value - Manual Strategy: {portvals_MS.iloc[-1] * start_val}")
        print(f"Final Portfolio Value - Strategy learner: {portvals_SL.iloc[-1] * start_val}")
        print(f"Final Portfolio Value - Benchmark: {benchmark.iloc[-1] * start_val}")


def author():
    return 'agupta857'


if __name__ == '__main__':
    print('Author ', author())
    compare_strategies()

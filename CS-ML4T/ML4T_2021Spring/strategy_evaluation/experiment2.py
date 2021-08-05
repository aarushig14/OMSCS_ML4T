"""
Student Name: Aarushi Gupta (replace with your name)
GT User ID: agupta857 (replace with your User ID)
GT ID: 903633934 (replace with your GT ID)
"""

import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import StrategyLearner as sl
from marketsimcode import compute_portvals, assess_portfolio


def compare_impacts_strategy_learner(verbose=False):
    symbol = "JPM"
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    start_val = 100000
    impacts = [0.05, 0.025, 0.01, 0.005, 0.0025, 0.0005]
    commission = 0.0

    results = [[], [], [], [], []]

    plt.figure(figsize=(8, 6))
    for i in impacts:
        learner = sl.StrategyLearner(verbose=False, commission=commission, impact=i)
        learner.add_evidence(symbol, start_date, end_date, start_val)
        sl_trades = learner.testPolicy(symbol, start_date, end_date, start_val)
        portvals = compute_portvals(symbol, sl_trades, start_val, commission=commission, impact=i)

        portvals = portvals / portvals.iloc[0]
        cr, adr, std, sr = assess_portfolio(portvals)
        results[0].append(cr)
        results[1].append(adr)
        results[2].append(std)
        results[3].append(sr)
        results[4].append(portvals.iloc[-1] * start_val)

        plt.plot(portvals, label=str(i))

    plt.title('Portfolio Against Different Impact Values')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.savefig('experiment2.png')

    columns = ['Cumulative Returns', 'Average Daily Returns', 'Standard Deviation', 'Sharpe Ratio', 'Final Portfolio Value']
    results = np.round(results, 3)
    df = pd.DataFrame(results, index=columns, columns=impacts)
    df.to_csv('impacts_sl.csv')

    if verbose:
        print('impacts - ', impacts)
        print('Results - ')
        print(df)
        print('Results saved to impacts_sl.csv and experiment2.png')


def author():
    return 'agupta857'


if __name__ == '__main__':
    print('Author ', author())
    compare_impacts_strategy_learner()

"""
Student Name: Aarushi Gupta (replace with your name)
GT User ID: agupta857 (replace with your User ID)
GT ID: 903633934 (replace with your GT ID)
"""

import StrategyLearner as sl
import ManualStrategy as ms
import experiment1
import experiment2

import datetime as dt

from indicators import getCombinedIndicators


def author():
    return 'agupta857'


if __name__ == "__main__":
    print('Author', author())
    symbol = "JPM"
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    start_val = 100000
    commission = 9.95
    impact = 0.005
    verbose = False

    print('------ Indicators Chart ------')
    getCombinedIndicators([symbol], start_date, end_date, lookback=20)

    print('------ Manual Strategy ------')
    ms.testStrategy(symbol, start_date, end_date, start_val, name='Portvals_MS_Insample.png', verbose=verbose)
    ms.testStrategy(symbol, dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31), start_val,
                 name='Portvals_MS_Outsample.png', verbose=verbose)

    print()
    print('------ Strategy Learner ------ \n------ BagLearner with Random Tree Learner ------')
    learner = sl.StrategyLearner(verbose=verbose, commission=commission, impact=impact)
    learner.add_evidence(symbol, start_date, end_date, start_val)
    learner.testStrategy(symbol, start_date, end_date, start_val)

    print()
    print('------ Experiment 1 ------')
    experiment1.compare_strategies(verbose)

    print()
    print('------ Experiment 2 ------')
    experiment2.compare_impacts_strategy_learner(verbose)

    print()
    print('Done !!')






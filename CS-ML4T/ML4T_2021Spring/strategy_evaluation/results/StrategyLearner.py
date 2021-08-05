"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Atlanta, Georgia 30332  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
All Rights Reserved  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Template code for CS 4646/7646  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
works, including solutions to the projects assigned in this course. Students  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
and other users of this template code are advised not to share it with others  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
or to make it available on publicly viewable websites including repositories  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
or edited.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
We do grant permission to share solutions privately with non-students such  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
as potential employers. However, sharing with other current or future  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
GT honor code violation.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
-----do not edit anything above this line---  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Student Name: Aarushi Gupta (replace with your name)
GT User ID: agupta857 (replace with your User ID)
GT ID: 903633934 (replace with your GT ID)
"""

import datetime as dt
import random
from ManualStrategy import get_benchmark
from indicators import rel_strength_idx, bollinger_band_value, ma_conv_div, exp_mov_avg
import pandas as pd
import util as ut
import RTLearner as rt
import BagLearner as bl
import numpy as np

from marketsimcode import compute_portvals, assess_portfolio
import matplotlib.pyplot as plt


class StrategyLearner(object):
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        If verbose = False your code should not generate ANY output.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type verbose: bool  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type impact: float  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param commission: The commission amount charged, defaults to 0.0  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type commission: float  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """

    # constructor
    def __init__(self, verbose=False, impact=9.95, commission=0.005):
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        Constructor method  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """
        seed = 8791649 # 11343179, 12305882, 715050410
        np.random.seed(seed)
        random.seed(seed)
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 5}, bags=25, boost=False, verbose=verbose)

    # this method should create a QLearner, and train it for trading
    def add_evidence(
            self,
            symbol="IBM",
            sd=dt.datetime(2008, 1, 1),
            ed=dt.datetime(2009, 1, 1),
            sv=10000,
    ):
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        Trains your strategy learner over a given time frame.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :param symbol: The stock symbol to train on  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :type symbol: str  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :type sd: datetime  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :type ed: datetime  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :param sv: The starting value of the portfolio  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :type sv: int  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """

        # add your code to do learning here
        dates = pd.date_range(sd, ed)
        prices = ut.get_data([symbol], dates)
        prices = pd.DataFrame(prices[symbol])
        normed_prices = prices / prices.iloc[0]

        rsi = rel_strength_idx(normed_prices, lookback=20)
        bbvalue = bollinger_band_value(normed_prices, lookback=20)
        macd = ma_conv_div(normed_prices)
        signal = exp_mov_avg(macd, lookback=9)

        N = 3

        indicators = pd.concat((rsi.rename(columns={symbol: 'RSI'}),
                                macd.rename(columns={symbol: 'MACD'}),
                                signal.rename(columns={symbol: 'SIGNAL'}),
                                bbvalue.rename(columns={symbol: 'BBVALUE'})),
                               axis=1)
        indicators.fillna(0, inplace=True)
        indicators=indicators[:-1*N]
        trainX = indicators.values

        trainY = []
        for i in range(normed_prices.shape[0] - N):
            ret = (normed_prices.iloc[i + N] / normed_prices.iloc[i]) - 1.0
            if ret[0] > 0.02:
                trainY.append(1)  # LONG
            elif ret[0] < -(0.02):
                trainY.append(-1)  # SHORT
            else:
                trainY.append(0)  # CASH
        trainY = np.array(trainY)

        # Training
        self.learner.add_evidence(trainX, trainY)

    # this method should use the existing policy and test it against new data  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    def testPolicy(
            self,
            symbol="IBM",
            sd=dt.datetime(2009, 1, 1),
            ed=dt.datetime(2010, 1, 1),
            sv=10000,
    ):
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        Tests your learner using data outside of the training data  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :param symbol: The stock symbol that you trained on on  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :type symbol: str  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :type sd: datetime  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :type ed: datetime  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :param sv: The starting value of the portfolio  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :type sv: int  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :rtype: pandas.DataFrame  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """
        dates = pd.date_range(sd, ed)
        prices = ut.get_data([symbol], dates)
        prices = pd.DataFrame(prices[symbol])
        normed_prices = prices / prices.iloc[0]

        rsi = rel_strength_idx(normed_prices, lookback=20)
        bbvalue = bollinger_band_value(normed_prices, lookback=20)
        macd = ma_conv_div(normed_prices)
        signal = exp_mov_avg(macd, lookback=9)

        indicators = pd.concat((rsi.rename(columns={symbol: 'RSI'}),
                                macd.rename(columns={symbol: 'MACD'}),
                                signal.rename(columns={symbol: 'SIGNAL'}),
                                bbvalue.rename(columns={symbol: 'BBVALUE'})),
                               axis=1)
        indicators.fillna(0, inplace=True)
        testX = indicators.values

        testY = self.learner.query(testX)
        testY = np.round(testY)

        orders = []
        date = []
        total = 0

        for i in range(testY.shape[0] - 1):
            buy = testY[i] == 1
            sell = testY[i] == -1

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

        trades = pd.DataFrame(data=orders, index=date)

        if self.verbose:
            print(type(trades))  # it better be a DataFrame!
        if self.verbose:
            print(trades)

        return trades

    def testStrategy(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
        trades = self.testPolicy(symbol, sd, ed, sv)
        benchmark = get_benchmark(symbol, sd, ed, sv)
        portvals = compute_portvals(symbol, trades, start_val, commission=9.95, impact=0.005)

        portvals = portvals / portvals.iloc[0]
        benchmark = benchmark / benchmark.iloc[0]

        plt.figure(figsize=(8, 6))
        plt.plot(portvals, label='Portfolio', color='red')
        plt.plot(benchmark, label='Benchmark', color='green')
        plt.title('JPM')
        plt.xlabel('Date')
        plt.ylabel('Nomralized Price')
        plt.legend()
        plt.savefig('Portvals_SL.png')

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

    def author(self):
        return 'agupta857'


if __name__ == "__main__":
    print("One does not simply think up a strategy")
    symbol = "JPM"
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    start_val = 100000

    learner = StrategyLearner(verbose=False)
    learner.add_evidence(symbol, start_date, end_date, start_val)
    learner.testStrategy(symbol, start_date, end_date, start_val)

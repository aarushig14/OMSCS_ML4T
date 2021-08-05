""""""

"""MC1-P2: Optimize a portfolio.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from util import get_data, plot_data
import scipy.optimize as spo

# GLOBAL VARIABLES
k = 252.0
start_val = 1000000
daily_rf = 0.0


def get_port_val(allocs, prices):
    normed = np.divide(prices, prices.iloc[0])
    alloced = normed * allocs
    pos_vals = alloced * start_val
    port_val = pos_vals.sum(axis=1)

    return port_val


def assess_portfolio(allocs, prices):
    port_val = get_port_val(allocs, prices)
    daily_returns = port_val.copy()
    daily_returns = (daily_returns / daily_returns.shift(1)) - 1
    daily_returns.iloc[0] = 0

    SR = (np.sqrt(k) * np.mean(daily_returns - daily_rf)) / np.std(daily_returns - daily_rf)
    cr, adr, sddr, sr = [
        np.divide(port_val.iloc[-1], port_val.iloc[0]) - 1,
        np.mean(daily_returns[1:]),
        np.std(daily_returns[1:]),
        SR
    ]
    return cr, adr, sddr, sr


def f(allocs, prices):
    _, _, _, SR = assess_portfolio(allocs, prices)
    return (-1) * SR


# This is the function that will be tested by the autograder  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
# The student must update this code to properly implement the functionality  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
def optimize_portfolio(
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 1, 1),
        syms=["GOOG", "AAPL", "GLD", "XOM"],
        gen_plot=False,
):
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    statistics.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type sd: datetime  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type ed: datetime  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        symbol in the data directory)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type syms: list  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        code with gen_plot = False.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type gen_plot: bool  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        standard deviation of daily returns, and Sharpe ratio  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :rtype: tuple  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """

    # Read in adjusted closing prices for given symbols, date range  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms,
                          dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later

    # find the allocations for the optimal portfolio

    n = len(syms)
    allocs = [1 / n] * n
    result = spo.minimize(f, allocs, args=prices,
                          method='SLSQP',
                          bounds=[(0, 1)] * n,
                          constraints=({'type': 'eq', 'fun': lambda x: 1.0 - np.sum(x)}))
    optimum_allocs = result.x

    # note that the values here ARE NOT meant to be correct for a test case
    cr, adr, sddr, sr = assess_portfolio(optimum_allocs, prices)  # add code here to compute stats

    # Get daily portfolio value  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    port_val = get_port_val(optimum_allocs, prices)  # add code here to compute daily portfolio values

    # Compare daily portfolio value with SPY using a normalized plot  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    if gen_plot:
        # add code to plot here  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        df_temp = pd.concat(
            [port_val, prices_SPY], keys=["Portfolio", "SPY"], axis=1
        )
        #
        df_temp = np.divide(df_temp, df_temp.iloc[0].values)
        # df_temp.iloc[0, :] = 0
        plt.figure(1)
        ax = df_temp.plot(title='Portfolio Value ' + str(syms) + ' and SPY')
        ax.set_ylabel('Normalized Prices')
        ax.set_xlabel('Dates')
        plt.savefig('plot.png')
        pass

    return optimum_allocs, cr, adr, sddr, sr


def test_code():
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    This function WILL NOT be called by the auto grader.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """

    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ['IBM', 'X', 'GLD', 'JPM']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )

    # Print statistics
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Symbols: {symbols}")
    print(f"Allocations:{allocations}")
    print(f"Sharpe Ratio: {sr}")
    print(f"Volatility (stdev of daily returns): {sddr}")
    print(f"Average Daily Return: {adr}")
    print(f"Cumulative Return: {cr}")


if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # Do not assume that it will be called  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    test_code()

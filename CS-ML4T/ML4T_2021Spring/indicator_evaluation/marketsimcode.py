""""""
"""MC2-P1: Market simulator.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
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
import os

import numpy as np

import pandas as pd
from util import get_data, plot_data
import matplotlib.pyplot as plt


def author():
    return 'agupta857'


def assess_portfolio(port_val):
    k = 252.0
    daily_rf = 0.0

    daily_returns = port_val.copy()
    daily_returns = daily_returns / daily_returns.shift(1) - 1

    SR = np.sqrt(k) * ((daily_returns - daily_rf).mean() / (daily_returns - daily_rf).std())
    cr, adr, sddr, sr = [
        np.divide(port_val.iloc[-1], port_val.iloc[0]) - 1,
        daily_returns.mean(),
        daily_returns.std(),
        SR
    ]

    return cr, adr, sddr, sr


def compute_portvals(
        orders,
        start_val=1000000,
        commission=9.95,
        impact=0.005,
):
    """
    Computes the portfolio values.

    :param orders_file: Path of the order file or the file object
    :type orders_file: str or file object
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input

    # Read orders
    orders.sort_index(ascending=True, inplace=True)

    start_date = orders.index.min()
    end_date = orders.index.max()

    symbols = np.unique(orders['Symbol'])
    columns = np.append(symbols, 'Cash')

    # Prices - [Date, Symbol1, Symbol2, ..., Cash]
    dates = pd.date_range(start_date, end_date)
    prices = get_data(symbols, dates)
    prices = prices[symbols]
    prices['Cash'] = 1

    # Trades - [Date, Symbol1, Symbol2, ..., Cash] - captures changes for every day
    trades = pd.DataFrame(index=prices.index, columns=columns, data=np.zeros(prices.shape))

    # Populate "trades" dataframe
    for date, row in orders.iterrows():
        sym, order, shares = row
        price = prices.loc[date, sym]
        if order == 'BUY':
            trades.loc[date, sym] += shares
            trades.loc[date, 'Cash'] -= price * shares
        else:
            trades.loc[date, sym] -= shares
            trades.loc[date, 'Cash'] += price * shares

        # substract transactional cost from cash
        trades.loc[date, 'Cash'] -= (commission + impact * price * shares)

    # populate 'Holdings' - [Date, Symbol1, Symbol2, ..., Cash] - captures everyday holdings
    holdings = trades.copy()
    holdings.iloc[0, -1] += start_val
    holdings = holdings.cumsum()

    # populate 'Values' - Prices * Holdings
    values = holdings * prices

    # calculate portfolio values = (prices * shares + cash) for each day
    portvals = values.sum(axis=1)
    return portvals


def marketsim(orders, sv=1000000):
    """
    Helper function to test code
    """
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters
    if not isinstance(orders, pd.DataFrame):
        "warning, orders should be a Dataframe."
        return None

    # Process orders
    portvals = compute_portvals(orders_file=orders, start_val=sv)

    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[
            0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = portvals.index.min()
    end_date = portvals.index.max()

    # save plot for portfolio values in the duration
    plt.figure(1)
    portvals.plot(title="portfolio")
    plt.savefig('portvals.png')

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = assess_portfolio(portvals)

    dates = pd.date_range(start_date, end_date)
    prices = get_data(['$SPX'], dates)
    prices = prices.loc[:, '$SPX']

    cum_ret_SPX, avg_daily_ret_SPX, std_daily_ret_SPX, sharpe_ratio_SPX = assess_portfolio(prices)

    # Compare portfolio against $SPX
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPX : {sharpe_ratio_SPX}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print(f"Cumulative Return of SPX : {cum_ret_SPX}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of SPX : {std_daily_ret_SPX}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of SPX : {avg_daily_ret_SPX}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")

    return portvals


if __name__ == "__main__":
    marketsim()

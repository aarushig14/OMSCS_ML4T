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
        symbol,
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

    symbols = [symbol]
    columns = np.append(symbols, 'Cash')

    # Prices - [Date, Symbol1, Symbol2, ..., Cash]
    dates = pd.date_range(start_date, end_date)
    prices = get_data(symbols, dates)
    prices = pd.DataFrame(prices[symbols])
    prices['Cash'] = 1

    # Trades - [Date, Symbol1, Symbol2, ..., Cash] - captures changes for every day
    trades = pd.DataFrame(index=prices.index, columns=columns, data=np.zeros(prices.shape))

    # Populate "trades" dataframe
    for date, row in orders.iterrows():
        order = row.values
        shares = abs(order)
        price = prices.loc[date, symbol]
        if order > 0:
            trades.loc[date, symbol] += shares
            trades.loc[date, 'Cash'] -= price * shares
        else:
            trades.loc[date, symbol] -= shares
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

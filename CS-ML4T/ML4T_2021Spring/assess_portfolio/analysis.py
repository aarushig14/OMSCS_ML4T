"""Analyze a portfolio.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Copyright 2017, Georgia Tech Research Corporation  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Atlanta, Georgia 30332-0415  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
All Rights Reserved  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import datetime as dt  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import numpy as np  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import pandas as pd  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
from util import get_data, plot_data
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
# This is the function that will be tested by the autograder  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
# The student must update this code to properly implement the functionality  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
def assess_portfolio(allocs, prices):
    sv = 1000000.0
    rfr = 0.0
    sf = 252.0
    norm_prices = prices * allocs * sv
    port_val = norm_prices.sum(axis=1)

    period_end = port_val.iloc[-1]
    commul = port_val.pct_change()
    cr = (period_end - sv) / sv
    adr = commul[1:].mean()
    sddr = commul[1:].std()
    sr = np.sqrt(sf) * (commul[1:] - rfr).mean() / ((commul[1:] - rfr).std())
    return [cr, adr, sddr, sr, port_val]
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
def test_code():  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    Performs a test of your code and prints the results  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # This code WILL NOT be tested by the auto grader  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # It is only here to help you set up and test your code  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # Define input parameters  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # Note that ALL of these values will be set to different values by  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # the autograder!  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    start_date = dt.datetime(2004, 12, 1)
    end_date = dt.datetime(2006, 5, 31)
    symbols = ["YHOO", "XOM", "GLD", "HNZ"]
    allocations = [0.0, 0.07, 0.59, 0.34]
    start_val = 1000000  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    risk_free_rate = 0.0  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    sample_freq = 252

    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(symbols,
                          dates)  # automatically adds SPY
    prices = prices_all[symbols]  # only portfolio symbols
    prices_SPY = prices_all["SPY"]
    prices = np.divide(prices, prices.iloc[0].values)


    # Assess the portfolio  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    cr, adr, sddr, sr, ev = assess_portfolio(
        allocs=allocations,
        prices=prices
    )


  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # Print statistics  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Start Date: {start_date}")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"End Date: {end_date}")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Symbols: {symbols}")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Allocations: {allocations}")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Sharpe Ratio: {sr}")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Average Daily Return: {adr}")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print(f"Cumulative Return: {cr}")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
if __name__ == "__main__":  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    test_code()  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			

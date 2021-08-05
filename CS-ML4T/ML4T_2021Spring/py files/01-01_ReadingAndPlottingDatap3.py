"""
(c) 2015 by Devpriya Dave and Tucker Balch.
"""

"""=================================================================================="""

"""Reading in a CSV File"""

import pandas as pd

def test_run():
    """Function called by Test Run."""
    df = pd.read_csv("data/AAPL.csv")
    # Quiz: Print last 5 rows of the data frame
    # print df				# prints entire data set (dataframe)
    # print df.head()		# prints first five records
    print (df.tail())			# prints last five records
	
if __name__ == "__main__":
    test_run()
	
"""=================================================================================="""

"""Select Rows"""

import pandas as pd

def test_run():
    """Function called by Test Run."""
    df = pd.read_csv("data/AAPL.csv")
    print (df[10:21])		# print rows between index 10 and 20 inclusive
	
if __name__ == "__main__":
    test_run()
	
"""=================================================================================="""

"""Computing Max Closing Price"""

import pandas as pd

def get_max_close (symbol):
    """Return the maximum closing value for stock indicated by symbol.

    Note: Data for a stock is stored in file: data/<symbol>.csv
    """
    df = pd.read_csv("data/{}.csv".format(symbol))  # read in data
    return  df['Close'].max() # compute and return max

def test_run():
    """Function called by Test Run."""
    for symbol in ['AAPL', 'IBM']:
        print ("Max close")
        print (symbol, get_max_close(symbol))


if __name__ == "__main__": # if run standalone
    test_run()
	
"""=================================================================================="""

"""Compute Mean Volume"""

import pandas as pd
import matplotlib.pyplot as plt
	
def get_mean_volume(symbol):
    """Return the mean volume for stock indicated by symbol.
    Note: Data for a stock is stored in file: data/<symbol>.csv
    """
    df = pd.read_csv("data/{}.csv".format(symbol))  # read in data
    # Quiz: Compute and return the mean volume for this stock
    return df['Volume'].mean()
   
def test_run():
    """Function called by Test Run."""
    for symbol in ['AAPL', 'IBM']:
        print ("Mean Volume")
        print (symbol, get_mean_volume(symbol))

		
if __name__ == "__main__":
    test_run()
	
"""=================================================================================="""

"""Plotting Stock Price Data"""

import pandas as pd
import matplotlib.pyplot as plt

def test_run():
    """Plot a single column."""
    df = pd.read_csv("data/AAPL.csv")
    print (df['Adj Close'])
    df['Adj Close'].plot()
    plt.show()  # must be called to show plots


if __name__ == "__main__":
    test_run()
	
"""=================================================================================="""

"""Plot High Prices for IBM"""

import pandas as pd
import matplotlib.pyplot as plt

def test_run():
    df = pd.read_csv("data/IBM.csv")
    # Quiz: Your code here
    print (df['High'])
    df['High'].plot()
    plt.show()  # must be called to show plots


if __name__ == "__main__":
    test_run()
	
"""=================================================================================="""

"""Plot Two Columns"""

import pandas as pd
import matplotlib.pyplot as plt
		
def test_run():
    df = pd.read_csv("data/IBM.csv")
    df[['Close', 'Adj Close']].plot()
    plt.show()  # must be called to show plots
	


if __name__ == "__main__":
    test_run()


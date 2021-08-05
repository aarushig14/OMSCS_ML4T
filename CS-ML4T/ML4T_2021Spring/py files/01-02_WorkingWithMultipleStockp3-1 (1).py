"""
(c) 2015 by Devpriya Dave and Tucker Balch.
"""

"""=================================================================================="""

"""Build a DataFrame in Pandas"""

import pandas as pd

def test_run():
	# Define data range
	start_date = '2010-01-22'
	end_date = '2010-01-26';
	dates=pd.date_range(start_date,end_date)
	
	print (dates)
	print (dates[0])  # get first element of list
	
	# Create an empty dataframe
	df1=pd.DataFrame(index=dates)  # define empty dataframe with these dates as index

	print (df1)
	
	# Read SPY data into temporary dataframe
	# dfSPY = pd.read_csv("data/SPY.csv") # will result in no data because this has index of integers
	# dfSPY = pd.read_csv("data/SPY.csv", index_col="Date", parse_dates=True)
	dfSPY = pd.read_csv("data/SPY.csv", index_col="Date",
						parse_dates=True, usecols=['Date','Adj Close'],
						na_values=['nan'])
	print (dfSPY)
	
	# Join the two dataframes using DataFram.join()
	df1=df1.join(dfSPY)
	print (df1)
	
	# Drop NaN Values
	df1 = df1.dropna()
	print (df1)

if __name__ == "__main__":
    test_run()

"""==========================================================================================================="""

"""Read in More Stocks"""

import pandas as pd

def test_run():
    # Define data range
    start_date = '2010-01-22'
    end_date = '2010-01-26';
    dates = pd.date_range(start_date, end_date)

    # Create an empty dataframe
    df1 = pd.DataFrame(index=dates)  # define empty dataframe with these dates as index

    # Read SPY data into temporary dataframe
    dfSPY = pd.read_csv("data/SPY.csv", index_col="Date",
                        parse_dates=True, usecols=['Date', 'Adj Close'],
                        na_values=['nan'])

    # Rename 'Adj Close' column to 'SPY' to prevent clash
    dfSPY = dfSPY.rename(columns={'Adj Close': 'SPY'})

    # Join the two dataframes using DataFram.join()
    df1 = df1.join(dfSPY, how='inner')

    # Read in more stocks
    symbols = ['GOOG', 'IBM', 'GLD']
    for symbol in symbols:
        df_temp = pd.read_csv("data/{}.csv".format(symbol, index_col='Date',
                                                   parse_dates=True, usecols=['Date', 'Adj Close'],
                                                   na_values=['nan']))
        df = df1.join(df_temp)  # use default how='left'

    print (df1)

if __name__ == "__main__":
    test_run()

"""==========================================================================================================="""

"""Utility functions"""

import os
import pandas as pd

def symbol_to_path(symbol, base_dir="data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
		# Quiz: Read and join data for each symbol
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df
	
"""==========================================================================================================="""

"""Slicing"""

import os
import pandas as pd

def symbol_to_path(symbol, base_dir="data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
		# Quiz: Read and join data for each symbol
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])
    return df

def test_run():
    # Define a date range
    dates = pd.date_range('2010-01-01', '2010-12-31')

    # Choose stock symbols to read
    symbols = ['GOOG', 'IBM', 'GLD']  # SPY will be added in get_data()

    # Get stock data
    df = get_data(symbols, dates)

	# Slice by row range (dates) using DataFram.ix[] selector
    print (df.ix['2010-01-01':'2010-01-31'])  # the month of January

    #  Slice by column (symbols)
    print (df['GOOG']) # a single label selects a single column
    print (df[['IBM', 'GLD']]) # a list of labels selects multiple columns

    # Slice by row and column
    print (df.ix['2010-03-01':'2010-03-15', ['SPY', 'IBM']])


if __name__ == "__main__":
    test_run()

"""==========================================================================================================="""

"""Plotting Multiple Stocks"""

import os
import pandas as pd
import matplotlib.pyplot as plt


def symbol_to_path(symbol, base_dir="data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
		# Quiz: Read and join data for each symbol
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df

#def plot_data(df):
#    """Plot stock prices."""
#	df.plot()
#    plt.show()

def plot_data(df, title="Stock prices"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()
	
def test_run():
	# Define a date range
    dates = pd.date_range('2010-01-01', '2010-12-31')

    # Choose stock symbols to read
    symbols = ['GOOG', 'IBM', 'GLD']  # SPY will be added in get_data()
    
    # Get stock data
    df = get_data(symbols, dates)

    # Plot
    plot_data(df)

if __name__ == "__main__":
    test_run()
	
"""==========================================================================================================="""

"""Slice and plot"""

import os
import pandas as pd
import matplotlib.pyplot as plt


def symbol_to_path(symbol, base_dir="data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
		# Quiz: Read and join data for each symbol
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df	

def normalize_data(df):
	"""Normalize stock prices using the first row of the dataframe."""
	return df/ df.ix[0,:] 
	
def plot_selected(df, columns, start_index, end_index):
    """Plot the desired columns over index values in the given range."""
    # Quiz: Your code here
    plot_data(df.ix[start_index:end_index,columns], title="Selected data")
	
def plot_data(df, title="Stock prices"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()


def test_run():
    # Define a date range
    dates = pd.date_range('2010-01-01', '2010-12-31')

    # Choose stock symbols to read
    symbols = ['GOOG', 'IBM', 'GLD']  # SPY will be added in get_data()
    
    # Get stock data
    df = get_data(symbols, dates)
	
    print (df)

    # Slice and plot
    plot_selected(df, ['SPY', 'IBM'], '2010-03-01', '2010-04-01')


if __name__ == "__main__":
    test_run()

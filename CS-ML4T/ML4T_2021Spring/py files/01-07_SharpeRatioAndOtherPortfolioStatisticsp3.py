"""
(c) 2015 by Devpriya Dave and Tucker Balch.
"""

"""=================================================================================="""


start_val = 1000000
start_date = 2009-1-1
end_date = 2011-12-31
symbols = ['SPY', 'XOM', 'GOOG', 'GLD']
allocs = [0.4, 0.4, 0.1, 0.1]


normed = prices/prices[0]
alloced = normed * allocs
pos_vals = alloced * start_val
port_val = pos_vals.sum(axis=1)

daily_rets = daily_rets[1:]
cum_ret = (port_val[-1]/port_val[0] - 1)
avg_daily_ret = daily_rets.mean()
std_daily_ret = daily_rets.std()

SR = sqrt(k) * mean(daily_rets - daily_rf) / std(daily_rets)




Environment -
-------------------------

Install and setup conda environment provided for ml4t.
http://lucylabs.gatech.edu/ml4t/spring2021/local-environment/
>> conda activate ml4t

Two python code files -
-------------------------

1. indicators.py 
	- Run python indicators.py
	- hit main function to execute the code and generate all plots shared in report for section 3

2. marketsimcode.py - computes portfolio using following API
	portvals = compute_portvals(orders, sv, 0.0, 0.0) # orders in dataframe

3. TheoreticallyOptimalStrategy.py 
	- To run again hit main function or execute from terminal
	>> python TheoreticallyOptimalStrategy.py
	- Use following api to generate orders
	testPolicy(symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)

4. testproject.py - Frontline function provided to run all the required api to generate the results shared in report using default values required.
	>> python testproject.py
	- hit main function from IDE.

To modify the input parameters enter values for variables in main function of testproject.py

How results are stored -
--------------------------

All graphs will be saved automatically in root folder.
Stats will be printed in console output.

Environment -
-------------------------

Install and setup conda environment provided for ml4t.
http://lucylabs.gatech.edu/ml4t/spring2021/local-environment/
>> conda activate ml4t

Two python code files -
-------------------------

1. indicators.py
	- Run python indicators.py to generate the combined indicators graph for in sample period.
	- hit main function to execute the code and generate all plots shared in report for section 3
	- getCombinedIndicators() generate the graphs.
	- Indicators API - bollinger_band_value(prices, 20), ma_conv_div(prices), rel_strength_idx(prices, 14), exp_mov_avg(macd, 9) - BBValue, MACD, RSI, Signal.

2. marketsimcode.py - computes portfolio using following API
	portvals = compute_portvals(symbol, orders, sv, commission=9.95, impact=0.005) # orders in dataframe
	statistics = assess_portfolio(portvals)

3. ManualStrategy.py
	- To run again hit main function or execute from terminal
	>> python ManualStrategy.py
	- Use following api to generate orders
	testPolicy(symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)

4. StrategyLearner.py
    To run again hit main function or execute from terminal
	>> python StrategyLearner.py
	- Use following api to generate orders
	learner = sl.StrategyLearner(verbose=verbose, commission=commission, impact=impact)
    learner.add_evidence(symbol, start_date, end_date, start_val)
    learner.testStrategy(symbol, start_date, end_date, start_val)

5. testproject.py - Frontline function provided to run all the required api to generate the results shared in report using default values required.
	>> python testproject.py
	- hit main function from IDE.
	- verbose = True : to print log statements
	- will generate all the graphs presented in report.

6. BagLearner.py and RTLearner.py - previous implementation imported to implement StrategyLearner.py

7. experiment1.py - execute and generate results for strategy comparison
    - compare_strategies() : api to call the experiment 1.

8. experiment2.py - execute and generate results for strategy learner with different impact values
    - compare_impacts_strategy_learner() : api to call the experiment 2.
    - modify the impacts array to test with different impact values.

To modify the input parameters tweak values for variables in main function of testproject.py

How results are stored -
--------------------------

All graphs will be saved automatically in root folder.
Stats will be printed in console output.
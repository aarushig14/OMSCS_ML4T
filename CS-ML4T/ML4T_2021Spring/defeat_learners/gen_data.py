""""""
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
template for generating data to fool learners (c) 2016 Tucker Balch  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


# this function should return a dataset (X and Y) that will work  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
# better for linear regression than decision trees  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
def best_4_lin_reg(seed=1489683273):
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param seed: The random seed for your data generation.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type seed: int  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :return: Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :rtype: numpy.ndarray  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """
    size = (100, 10)
    np.random.seed(seed)
    # contibution of 10 factors in deciding to buy a property
    X = np.random.randint(1, 100, size)

    # weights of each factor
    weights = np.random.random(10)

    # final price of the property
    Y = np.dot(X, weights)
    return X, Y


def best_4_dt(seed=1489683273):
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param seed: The random seed for your data generation.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type seed: int  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :return: Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :rtype: numpy.ndarray  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """
    size = (100, 6)
    np.random.seed(seed)

    # grade scored in 6 subjects
    X = np.random.choice([1, 2, 3, 4, 5], size, p=[0.05, 0.15, 0.35, 0.3, 0.15])

    # Y - 1: Science stream, 2: Medical Stream, 3: Finance Stream, 4: Arts Stream
    Y = np.zeros(size[0])
    for i in range(size[0]):
        if X[i, 0] >= 4:
            Y[i] = 1
        elif X[i, 1] >= 4:
            Y[i] = 2
        elif X[i, 2] >= 4:
            Y[i] = 3
        else:
            Y[i] = 4

    return X, Y


def author():
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :return: The GT username of the student  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :rtype: str  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """
    return "agupta857"  # Change this to your user ID


if __name__ == "__main__":
    X, Y = best_4_lin_reg()

    plt.figure(1)
    plt.scatter(X[:, 0], Y)
    plt.savefig("best_4_lin_reg.png")

    df = pd.DataFrame(np.column_stack((X, Y)))
    print(df.describe())

    X, Y = best_4_dt()

    plt.figure(2)
    plt.scatter(np.sum(X, axis=1), Y)
    plt.savefig("best_4_dt.png")

    df = pd.DataFrame(np.column_stack((X, Y)))
    print(df.describe())

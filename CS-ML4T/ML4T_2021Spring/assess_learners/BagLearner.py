""""""
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
A simple wrapper for Bag Learner.  (c) 2015 Tucker Balch  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			

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
"""

import numpy as np


class BagLearner(object):
    """
    This is a Bag Learner. It is implemented correctly.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

    def __init__(self, learner, kwargs={}, bags=10, boost=False, verbose=False):
        """
        Constructor method
        """
        learners = []
        for i in range(0, bags):
            learners.append(learner(**kwargs))
        self.learners = learners
        self.kwargs = kwargs
        self.verbose = verbose
        self.boost = boost
        self.bags = bags
        if self.verbose:
            print("Instantiation complete.")
            self.__verbose()
        # move along, these aren't the drones you're looking for

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "agupta857"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        data = np.column_stack((data_x, data_y[:, np.newaxis]))
        num_instances = data_x.shape[0]
        for learner in self.learners:
            rand_idx = np.random.choice(num_instances, num_instances, replace=True)
            learner.add_evidence(data_x[rand_idx], data_y[rand_idx])

        if self.verbose:
            print("Learning completed.")
            self.__verbose()

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        results = np.array([learner.query(points) for learner in self.learners])
        if self.verbose:
            print("Query completed.")
            self.__verbose()
        return np.mean(results, axis=0)

    def __verbose(self):
        print("Learner arguments: ", self.kwargs)
        print("Total number of bags: ", self.bags)
        print("Boosting: ", self.boost)
        print("Author: ", self.author())


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")

""""""
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
A simple wrapper for Insane Learner.  (c) 2015 Tucker Balch  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			

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
import BagLearner as bl
import LinRegLearner as lrl


class InsaneLearner(object):
    """
    This is a Insane Learner. It is implemented correctly.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

    def __init__(self, verbose=False, arbitrary_learner=lrl.LinRegLearner, bags=20, learn_instance=20, kwargs={},
                 boost=False):
        """
        Constructor method
        """
        self.learners = [bl.BagLearner(learner=arbitrary_learner, kwargs=kwargs, bags=learn_instance, verbose=verbose, boost=boost) for i in range(0, bags)]
        if verbose:
            print("Number of bags: {} with {} learning instance.",bags,learn_instance)

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
        for learner in self.learners:
            rand_idx = np.random.choice(data_x.shape[0], data_x.shape[0], replace=True)
            learner.add_evidence(data_x[rand_idx], data_y[rand_idx])

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        results = np.array([learner.query(points) for learner in self.learners])
        return np.mean(results, axis=0)


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")

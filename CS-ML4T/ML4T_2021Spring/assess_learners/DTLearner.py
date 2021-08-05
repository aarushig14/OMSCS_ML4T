""""""
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
A simple wrapper for Decision Tree Learner.  (c) 2015 Tucker Balch 		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			

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
from scipy import stats


class DTLearner(object):
    """
    This is a Decision Tree Learner. It is implemented correctly.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

    def __init__(self, leaf_size=1, verbose=False):
        """
        Constructor method
        """
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None
        if self.verbose:
            print("Instantiation complete.")
            self.__verbose()

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "agupta857"  # replace tb34 with your Georgia Tech username

    def __best_feature(self, data):
        num_features = data.shape[1] - 1
        correlations = np.zeros(num_features)
        for i in range(data.shape[1] - 1):
            cor = np.abs(stats.pearsonr(data[:, i], data[:, -1])[0])
            correlations[i] = 0.0 if np.isnan(cor) else cor

        attempt = 0
        while attempt < num_features:
            max_corr_feature = np.argmax(correlations)
            SplitVal = np.median(data[:, max_corr_feature])
            left_len = len(data[data[:, max_corr_feature] <= SplitVal])
            right_len = len(data[data[:, max_corr_feature] > SplitVal])

            if left_len > 0 and right_len > 0:
                return max_corr_feature

            correlations[max_corr_feature] = -1
            attempt += 1

        return -1

    def __build_tree(self, data):
        if data.shape[0] <= self.leaf_size:
            return np.array([[-1, np.mean(data[:, -1]), -1, -1]])
        unique_y = np.unique(data[:, -1])
        if unique_y.shape[0] == 1:
            return np.array([[-1, unique_y[0], -1, -1]])

        num_features = data.shape[1] - 1
        features_remaining = list(range(num_features))
        correlations = np.zeros(num_features)
        for i in range(data.shape[1] - 1):
            cor = np.abs(stats.pearsonr(data[:, i], data[:, -1])[0])
            correlations[i] = 0.0 if np.isnan(cor) else cor

        attempt = 0
        while len(features_remaining) > 0:
            best_feature = np.argmax(correlations)
            SplitVal = np.median(data[:, best_feature])
            left_len = data[data[:, best_feature] <= SplitVal].shape[0]
            right_len = data[data[:, best_feature] > SplitVal].shape[0]

            if left_len > 0 and right_len > 0:
                break

            correlations[best_feature] = -1
            features_remaining.remove(best_feature)
            attempt += 1

        if len(features_remaining) == 0:
            m = stats.mode(data[:, -1])
            return np.array([[-1, np.max(m[0]), -1, -1]])

        SplitVal = np.median(data[:, best_feature])
        lefttree = self.__build_tree(data[data[:, best_feature] <= SplitVal])
        righttree = self.__build_tree(data[data[:, best_feature] > SplitVal])
        root = [best_feature, SplitVal, 1, lefttree.shape[0] + 1]
        return np.vstack((root, lefttree, righttree))

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        data = np.column_stack((data_x, data_y[:, np.newaxis]))
        decision_tree = self.__build_tree(data)

        if self.tree == None:
            self.tree = decision_tree
        else:
            np.vstack((self.tree, decision_tree))

        if self.verbose:
            print("Learning completed.")
            self.__verbose()

    def __search_tree(self, point, node):
        row = self.tree[node]
        feature = int(row[0])
        splitval = row[1]
        if feature == -1:
            return splitval
        left_node = int(node + row[2])
        right_node = int(node + row[3])

        if point[feature] <= splitval:
            return self.__search_tree(point, left_node)
        else:
            return self.__search_tree(point, right_node)

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        pred_y = []
        for point in points:
            pred_val = self.__search_tree(point, 0)
            pred_y.append(pred_val)

        if self.verbose:
            print("Query completed.")
            self.__verbose()

        return pred_y

    def __verbose(self):
        print("Leaf size: ", self.leaf_size)
        print("Total number of nodes: ", self.tree.shape[0])
        print("Author: ", self.author())


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")

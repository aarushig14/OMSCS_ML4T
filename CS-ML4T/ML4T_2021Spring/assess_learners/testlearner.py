""""""
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Test a learner.  (c) 2015 Tucker Balch  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
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

import math
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import LinRegLearner as lrl
import DTLearner as dtl
import RTLearner as rtl
import BagLearner as bl
import InsaneLearner as il


def experiment_1(train_x, train_y, test_x, test_y, arbitrary_learner=dtl.DTLearner, low=1, high=100):
    insample_rmse=[]
    outsample_rmse = []
    leaf_sizes = list(range(low, high+1))
    for i in leaf_sizes:
        learner = arbitrary_learner(leaf_size=i, verbose=False)
        learner.add_evidence(train_x, train_y)

        pred_in = learner.query(train_x)
        insample_rmse.append(math.sqrt(((train_y - pred_in) ** 2).sum() / train_y.shape[0]))

        pred_out = learner.query(test_x)
        outsample_rmse.append(math.sqrt(((test_y - pred_out) ** 2).sum() / test_y.shape[0]))

    plt.figure(1)
    plt.plot(leaf_sizes, insample_rmse, color='tab:blue', label="insample")
    # plt.plot(leaf_sizes, outsample_rmse, color='tab:orange', label="outsample")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.title("RMSE InSample vs OutSample - Overfitting")
    plt.legend()
    plt.savefig("experiment1a.png")

    plt.figure(2)
    plt.plot(leaf_sizes, insample_rmse, color='tab:blue', label="insample")
    plt.plot(leaf_sizes, outsample_rmse, color='tab:orange', label="outsample")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.xlim(5, 12)
    plt.title("RMSE InSample vs OutSample - Overfitting")
    plt.legend()
    plt.savefig("experiment1b.png")


def experiment_2(train_x, train_y, test_x, test_y, arbitrary_learner=dtl.DTLearner, low=1, high=100, bags=20, **kwargs):
    insample_rmse=[]
    outsample_rmse = []
    leaf_sizes = list(range(low, high+1))
    np.random.seed(987654321)
    for i in leaf_sizes:
        learner = bl.BagLearner(learner=arbitrary_learner, kwargs={"leaf_size": i, **kwargs}, bags=bags, verbose=False)
        learner.add_evidence(train_x, train_y)

        pred_y = learner.query(train_x)
        insample_rmse.append(math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0]))

        pred_y = learner.query(test_x)
        outsample_rmse.append(math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0]))
    plt.figure(3)
    plt.plot(leaf_sizes, insample_rmse, color='tab:blue', label="insample")
    plt.plot(leaf_sizes, outsample_rmse, color='tab:orange', label="outsample")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.title("Effect of Bag Learner on overfitting.")
    plt.legend()
    plt.savefig("experiment2a.png")

    plt.figure(4)
    plt.plot(leaf_sizes, insample_rmse, color='tab:blue', label="insample")
    plt.plot(leaf_sizes, outsample_rmse, color='tab:orange', label="outsample")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.title("Effect of Bag Learner on overfitting.")
    plt.xlim(5, 12)
    plt.legend()
    plt.savefig("experiment2b.png")

def plot_time_taken(train_x, train_y, test_x, test_y, low=1, high=100):
    add_time_dt = []
    add_time_rt = []

    query_time_dt = []
    query_time_rt = []

    leaf_sizes = np.array(range(low, high + 1), dtype=np.int)
    np.random.seed(987654321)

    for i in leaf_sizes:
        dt = dtl.DTLearner(leaf_size=i, verbose=False)
        rt = rtl.RTLearner(leaf_size=i, verbose=False)

        t0 = time.time()
        dt.add_evidence(train_x, train_y)
        add_time_dt.append(np.round(time.time() - t0, 2))

        t0 = time.time()
        rt.add_evidence(train_x, train_y)
        add_time_rt.append(np.round(time.time() - t0, 2))

        t0 = time.time()*1000000000
        dt.query(test_x)
        query_time_dt.append(np.round(time.time()*1000000000 - t0, 2))

        t0 = time.time()*1000000000
        rt.query(test_x)
        query_time_rt.append(np.round(time.time()*1000000000 - t0, 2))

    print("Avg training time DTLearner: ", np.mean(add_time_dt))
    print("Avg training time RTLearner: ", np.mean(add_time_rt))
    print("Avg query time DTLearner: ", np.mean(query_time_dt))
    print("Avg query time RTLearner: ", np.mean(query_time_rt))

    plt.figure(5)
    plt.title("Training Time DTLearner vs RTLearner")
    plt.plot(leaf_sizes, add_time_dt, color='tab:blue', label="DTLearner")
    plt.plot(leaf_sizes, add_time_rt, color='tab:orange', label="RTLearner")
    plt.xlabel("Leaf Size")
    plt.ylabel("Time in Seconds")
    plt.legend()
    plt.savefig("experiment3a.png")

    plt.figure(6)
    plt.title("Query Time DTLearner vs RTLearner")
    plt.plot(leaf_sizes, query_time_dt, color='tab:blue', label="DTLearner")
    plt.plot(leaf_sizes, query_time_rt, color='tab:orange', label="RTLearner")
    plt.xlabel("Leaf Size")
    plt.ylabel("Time in Seconds")
    plt.legend()
    plt.savefig("experiment3b.png")


def plot_space(train_x, train_y, low=1, high=100):
    leaf_sizes = np.array(range(low, high + 1), dtype=np.int)
    size_dt = []
    size_rt = []

    np.random.seed(987654321)

    for i in leaf_sizes:
        dt = dtl.DTLearner(leaf_size=i, verbose=False)
        rt = rtl.RTLearner(leaf_size=i, verbose=False)

        dt.add_evidence(train_x, train_y)
        rt.add_evidence(train_x, train_y)

        size_dt.append(dt.tree.shape[0])
        size_rt.append(rt.tree.shape[0])

    print("Average number of nodes in DT: ", np.mean(size_dt))
    print("Average number of nodes in RT: ", np.mean(size_rt))

    plt.figure(7)
    plt.title("Space - DTLearner vs RTLearner")
    plt.plot(leaf_sizes, size_dt, color='tab:blue', label="DTLearner")
    plt.plot(leaf_sizes, size_rt, color='tab:orange', label="RTLearner")
    plt.xlabel("Leaf Size")
    plt.ylabel("Number of Nodes")
    plt.ylim(0, 300)
    plt.legend()
    plt.savefig("experiment3e.png")


def plot_precision(train_x, train_y, test_x, test_y, low=1, high=100, precision=0.5):
    leaf_sizes = np.array(range(low, high + 1), dtype=np.int)
    prec_in_dt = []
    prec_in_rt = []
    prec_out_dt = []
    prec_out_rt = []

    np.random.seed(987654321)

    for i in leaf_sizes:
        dt = dtl.DTLearner(leaf_size=i, verbose=False)
        rt = rtl.RTLearner(leaf_size=i, verbose=False)

        dt.add_evidence(train_x, train_y)
        rt.add_evidence(train_x, train_y)

        # in sample
        pred_y_dt = np.array(dt.query(train_x))
        pred_y_rt = np.array(rt.query(train_x))

        true_posi = abs(pred_y_dt - train_y)/train_y < precision
        prec_in_dt.append(np.count_nonzero(true_posi)/len(pred_y_dt))
        true_posi = abs(pred_y_rt - train_y)/train_y < precision
        prec_in_rt.append(np.count_nonzero(true_posi)/len(pred_y_rt))


        # out sample
        pred_y_dt = dt.query(test_x)
        pred_y_rt = rt.query(test_x)

        true_posi = abs(pred_y_dt - test_y)/test_y < precision
        prec_out_dt.append(np.count_nonzero(true_posi)/len(pred_y_dt))
        true_posi = abs(pred_y_rt - test_y) / test_y < precision
        prec_out_rt.append(np.count_nonzero(true_posi)/len(pred_y_rt))

    print("Avg precision count in-sample DTLearner: ", np.mean(prec_in_dt))
    print("Avg precision count in-sample RTLearner: ", np.mean(prec_in_rt))
    print("Avg precision count out-sample DTLearner: ", np.mean(prec_out_dt))
    print("Avg precision count out-sample RTLearner: ", np.mean(prec_out_rt))

    plt.figure(8)
    plt.title("InSample Precision - DTLearner vs RTLearner")
    plt.plot(leaf_sizes, prec_in_dt, color='tab:blue', label="DTLearner")
    plt.plot(leaf_sizes, prec_in_rt, color='tab:orange', label="RTLearner")
    plt.xlabel("Leaf Size")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig("experiment3c.png")

    plt.figure(9)
    plt.title("OutSample Precision - DTLearner vs RTLearner")
    plt.plot(leaf_sizes, prec_out_dt, color='tab:blue', label="DTLearner")
    plt.plot(leaf_sizes, prec_out_rt, color='tab:orange', label="RTLearner")
    plt.xlabel("Leaf Size")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig("experiment3d.png")


def experiment_3(train_x, train_y, test_x, test_y, low=1, high=100):
    plot_time_taken(train_x, train_y, test_x, test_y, low, high)
    plot_precision(train_x, train_y, test_x, test_y, low, high)
    plot_space(train_x, train_y)


if __name__ == "__main__":
    start = time.time()
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.argv[1] = "Data/Istanbul.csv"
        # sys.exit(1)
    inf = open(sys.argv[1])
    data = []
    for s in inf.readlines():
        row = s.strip().split(",")
        data_e = []
        for e in row:
            try:
                data_e.append(float(e))
            except:
                continue
        if len(data_e) > 0:
            data.append(data_e)

    data = np.array(data)
    # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    experiment_1(train_x, train_y, test_x, test_y)
    experiment_2(train_x, train_y, test_x, test_y)
    experiment_3(train_x, train_y, test_x, test_y)
    end = time.time()

    print("Time taken to execute: ", end-start, " seconds.")

    # print(test_x.shape)
    # print(test_y.shape)

    # # create a learner and train it
    # learner = dtl.DTLearner(leaf_size=1, verbose=False)
    # learner.add_evidence(train_x, train_y)  # train it
    # print(learner.author())
    #
    # # evaluate in sample
    # pred_y = learner.query(
    #     train_x)  # get the predictions
    # rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    # print()
    # print("In sample results")
    # print("RMSE: ", rmse)
    # c = np.corrcoef(pred_y, y=train_y)
    # print("corr: ", c[0, 1])
    #
    # # evaluate out of sample
    # pred_y = learner.query(test_x)  # get the predictions
    # rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    # print()
    # print("Out of sample results")
    # print("RMSE: ", rmse)
    # c = np.corrcoef(pred_y, y=test_y)
    # print("corr: ", c[0, 1])

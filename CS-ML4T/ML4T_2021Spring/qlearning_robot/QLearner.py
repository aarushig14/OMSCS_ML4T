""""""
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
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

import random as rand

import numpy as np


class QLearner(object):
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    This is a Q learner object.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param num_states: The number of states to consider.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type num_states: int  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param num_actions: The number of actions available..  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type num_actions: int  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type alpha: float  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type gamma: float  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type rar: float  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type radr: float  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type dyna: int  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type verbose: bool  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """

    def __init__(
            self,
            num_states=100,
            num_actions=4,
            alpha=0.2,
            gamma=0.9,
            rar=0.5,
            radr=0.99,
            dyna=0,
            verbose=False,
    ):
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        Constructor method  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """
        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0

        # initialising parameters
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose

        # Q-table
        self.Q = np.zeros((num_states, num_actions))

        # T and R
        self.T = np.zeros((num_states, num_actions, num_states))
        self.T_Count = self.T.copy()
        self.R = self.Q.copy()

    def querysetstate(self, s):
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        Update the state without updating the Q-table
        :param s: The new state  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :type s: int  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :return: The selected action  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :rtype: int  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """
        self.s = s

        # Random action
        action = rand.randint(0, self.num_actions - 1)

        # Update action if random probability not satisfied
        if np.random.random() >= self.rar:
            action = np.argmax(self.Q[s])

        if self.verbose:
            print(f"s = {s}, a = {action}")
        return action

    def query(self, s_prime, r):
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        Update the Q table and return an action
        :param s_prime: The new state  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :type s_prime: int  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :param r: The immediate reward  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :type r: float  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :return: The selected action  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        :rtype: int  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """

        s = self.s
        a = self.a

        # update Q table
        q_prime = (1 - self.alpha) * self.Q[s, a]
        future_reward = self.alpha * (r + (self.gamma * np.max(self.Q[s_prime])))
        self.Q[s, a] = q_prime + future_reward

        if self.dyna > 0:

            # build the model T and R
            self.T_Count[s, a, s_prime] += 1
            self.T[s, a, s_prime] = self.T_Count[s, a, s_prime] / np.sum(self.T_Count[s, a])
            self.R[s, a] = (1 - self.alpha) * self.R[s, a] + (self.alpha * r)

            # hallucinate
            for i in range(self.dyna):
                dyna_s = rand.randint(0, self.num_states - 1)
                dyna_a = rand.randint(0, self.num_actions - 1)
                dyna_s_prime = np.argmax(self.T[dyna_s, dyna_a])
                dyna_r = self.R[dyna_s, dyna_a]

                dyna_q_prime = (1 - self.alpha) * self.Q[dyna_s, dyna_a]
                syna_future_reward = self.alpha * (dyna_r + (self.gamma * np.max(self.Q[dyna_s_prime])))
                self.Q[dyna_s, dyna_a] = dyna_q_prime + syna_future_reward

        # Random action
        action = rand.randint(0, self.num_actions - 1)

        # Update action if random probability not satisfied
        if np.random.random() >= self.rar:
            action = np.argmax(self.Q[s_prime])

        # Update random action rate
        self.rar = self.rar * self.radr

        self.s = s_prime
        self.a = action

        # Print debug statements
        if self.verbose:
            print(f"s = {s}, a = {a}, s_prime={s_prime}, r={r}, a_prime={action}")
        return action

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "agupta857"


if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
    learner = QLearner()
    print(learner.author())

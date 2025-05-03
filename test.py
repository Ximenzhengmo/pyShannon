from markov import *
from utils import *

r"""
Test script for the MarkovChain class.

This script initializes a Markov chain with a given transition matrix and parameters
"""

__float_dtype__ = np.float64
np.set_printoptions(precision=4)

# Initializes a Markov chain with a given transition matrix and parameters.
mrkv1 = MarkovChain([0.25, 0.75, 0.6, 0.4, 0.9, 0.1, 0.2, 0.8], 2, 2)
mrkv1.setup(2, [0.2308, 0.1923, 0.1923, 0.3846])

# Calculates the joint probability distibution of a k-length chain
k = 4
p_k = mrkv1.prob_topk(k)
print(p_k)

# Calculates the joint probability distibution of ( X_1, X_2, X_5 )
k = [1, 2, 5]
p_k = mrkv1.prob_k(k)
print(p_k)
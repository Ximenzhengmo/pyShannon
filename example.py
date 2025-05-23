from chain import *
from source import *
from utils import *

r"""
Test script for the MarkovChain class.

This script initializes a Markov chain with a given transition matrix and parameters
"""

__float_dtype__ = np.float64
__check_cache_size__ = 16
np.set_printoptions(precision=4)

# Initializes a Markov chain with a given transition matrix and parameters.
mrkv1 = MarkovChain([0.25, 0.75, 0.6, 0.4, 0.9, 0.1, 0.2, 0.8], 2, 2)
mrkv1.setup(2, [0.2308, 0.1923, 0.1923, 0.3846])

# Calculates the joint probability distibution of a k-length chain
k = 4
p_k = mrkv1.prob_topk(k)
print(p_k)

# Calculates the joint probability distibution of ( X_1, X_2, X_5 )
k = [1, 2, 8]
p_k = mrkv1.prob_k(k)
print(p_k)

# Calculates the Entropy of p_k
print(Entropy(p_k.flatten()))

# Initializes a MemoryLessChain with a given probability distribution and parameters.
mlc = MemoryLessChain([0.25, 0.75], 2)

# Calculates the joint probability distibution of a k-length chain
k = 3
p_k = mlc.prob_topk(k)
print(p_k)

# create a 2-symbol MemoryLessSource with a given probability distribution and parameters.
# generate `n` random sequences of length `k` from the source.
s = Source([0.75, 0.25], symbol=['0', '1'])
itera = s.random_sequence_gen(9, prior='0-1-@-0-1-@-0', placeholder='@', sep='-')
for i in itera(5):
    print(i)

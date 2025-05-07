import numpy as np
import warnings
import functools
from utils import check_cache, deprecate, _prob_dstrbt_check, __float_dtype__, __check_cache_size__
warnings.filterwarnings("default")


class MemoryLessChain():
    r"""
    A class to represent a memoryless chain.

    Parameters:
        P (array_like): probability distribution ( elem_num = x ).
        x (int): Number of symbols.
        m (deprecated, useless): This is a deprecated parameter for param-list-align with Class `MarkovChain`, which is recommended to use for MemoryChain.
    """
    def __init__(self, P, x, m=None):
        r"""
        Initialize the memoryless chain with a probability distribution and symbol number.
        Parameters:
            P (array_like): probability distribution ( elem_num = x ).
            x (int): Number of symbols.
            m (deprecated, useless): This is a deprecated parameter for param-list-align with Class `MarkovChain`, which is recommended to use for MemoryChain.
        """
        try:
            P = np.asarray(P, dtype=__float_dtype__).reshape(x)  # Ensure P is a numpy array of floats
        except ValueError:
            raise ValueError("`P` must be an array-like of numbers with x**2 elements.")

        assert _prob_dstrbt_check(P), "Probabilities `P` must be in the range [0, 1] and sum(P) = [1, 1,..]."
        self.x = x
        self.P = P 

    def prob_topk(self, k):
        r"""
        Calculate the joint probability distribution of top-k symbol the memoryless chain.
        - P(X_1, X_2, ..., X_k)

        Parameters:
            k (int): the index ( 1 begin ) of the Markov Chain sequence.
        
        Returns:
            array_like: the joint probability distribution of top-k symbol the memoryless chain.
            shape = (x, x, ..., x) ( k times )
        """
        if k < 0:
            raise ValueError("`k` must be greater than or equal to 0.")
        if k == 0:
            warnings.warn("`0` index will be ignored. The probability distribution of X_0 is 1.")
            return 1.
        P = self.P.copy()
        for i in range(1, k):
            P = np.outer(P, self.P).flatten()
        return P.reshape(*( [self.x] * k ))

    def prob_k(self, k):
        r"""
        Calculate the probability distribution of k-th or joint probability distribution of X_k symbol the memoryless chain. For example, if k=2, return the probability distribution of X_2, if k=[1, 3], return the joint probability distribution of X_1,X_3.
        - k = list(range(1, k+1)) -> P(X_1, X_2, ..., X_k) equivalent to `self.prob_topk(k)`
        - k = [1, 3] -> P(X_1, X_3)
        - k = [1, 2, 3] -> P(X_1, X_2, X_3)

        Parameters:
            k (int or int_list ): the index ( 1 begin ) of the Markov Chain sequence.
        
        Returns:
            array_like: the probability distribution.
            shape = (x, x, ..., x) ( len(k) times )
        """
        if isinstance(k, int):
            k = [k]
        assert len(k) > 0, "k(list) can't be empty."
        assert np.all( map(lambda x: isinstance(x, int) and x >= 0, k) ), "k must be int or int_list and k >= 0."
        assert np.all( np.diff(k) > 0 ),  "the elem of k must be increasing strictly."
        if k[0] == 0:
            warnings.warn("`0` index will be ignored. The probability distribution of X_0 is 1.")
            k = k[1:]
        p = self.prob_topk(len(k))
        return p


class MarkovChain():
    r"""
    A class to represent a Markov Chain.

    Parameters:
        P_trans (array_like): m-order Transition probability matrix ( elem_num = x**(m+1) ).
        x (int): Number of symbols.
        m (int): Order of the Markov chain.
    """
    def __init__(self, P_trans, x : int, m : int):
        r"""
        Initialize the Markov Chain with a transition matrix and parameters. The 0-order markov chain is a special case of the 1-order Markov chain, it will be treated as a 1-order Markov chain. It is recommended to use `MemoryLessChain` for 0-order Markov chain.

        Parameters:
            P_trans (array_like): m-order Transition probability matrix ( elem_num = x**(m+1) ).
            x (int): Number of symbols.
            m (int): Order of the Markov chain.
        """
        try:
            P_trans = np.asarray(P_trans, dtype=__float_dtype__).reshape(x**m, x)  # Ensure P_m is a numpy array of floats
        except ValueError:
            raise ValueError("`P_m` must be an array-like of numbers with x**(m+1) elements.")

        assert _prob_dstrbt_check(P_trans, axis=-1), "Probabilities `P_m` must be in the range [0, 1] and sum(P_m, axis=-1) = [1, 1,..]."
        self.m = m # the order of the Markov chain
        self.x = x # the number of symbols in the source
        self.P_trans = None # the transition probability matrix ( m-order )
        self.P_trans_eq = None # The 1-order transition matrix of the equivalent first-order Markov chain
        if self.m == 0:
            self.m = 1
            self.P_base_m = P_trans[0]
            self.P_trans = np.concatenate([P_trans] * self.x, axis=0 ) 
            self.P_trans_eq = self.P_trans
        else:
            self.P_trans = P_trans
            self.P_trans_eq = np.zeros((self.x, self.x**(self.m-1), self.x**(self.m-1), self.x))
            for i in range(self.x):
                for j in range(self.x**(self.m-1)):
                    self.P_trans_eq[i, j, j, :] = self.P_trans[j + i * self.x**(self.m-1) ,:]
            self.P_trans_eq = self.P_trans_eq.reshape(self.x**self.m, self.x**self.m)


    def setup(self, k : int, P_k):
        r"""
        Setup the Markov Chain with a transition matrix and parameters. Call this function to set the beginning of the Markov Chain if the order of the Markov Chain greater than 0.

        Parameters:
            k (int): the length of the known Markov Chain.
            P_k (array_like): the probability distribution of the known Markov Chain ( elem_num = x**k ).
        """
        if k < self.m:
            raise ValueError("`k` must be greater than or equal to the order `m`.")
        try:
            P_k = np.asarray(P_k, dtype=__float_dtype__).reshape(*( [self.x] * self.m + [-1] ))  # Ensure P_m is a numpy array of floats
        except ValueError:
            raise ValueError("`P_m` must be an array-like of numbers with x**(m+1) elements.")
        assert _prob_dstrbt_check(P_k), "Probabilities `P` must be in the range [0, 1] and sum(P) = 1."
        self.P_base_m = P_k.sum(axis=-1)

    @check_cache
    def _step_matrix(self, d : int, newdim : int):
        assert d >= 0, "d must be greater than or equal to 0."
        assert newdim <= self.m, "newdim must be less than or equal to m."
        return np.linalg.matrix_power(self.P_trans_eq, d).reshape(self.x**self.m, self.x**(self.m-newdim), self.x**newdim).sum(axis=1)

    def _get_steps(self, k):
        import bisect
        steps = []
        n = len(k)
        i = 0
        while i < n:
            j = bisect.bisect_right(k, k[i] + self.m - 1) - 1
            steps.append(k[j])
            i = j + 1
        return steps

    def _prob_step(self, p, d, newdim):
        p_rsp = p.reshape(-1, self.x**self.m, 1)
        matrix = self._step_matrix(d, newdim).reshape(1, self.x**self.m, self.x**newdim)
        return np.multiply(matrix, p_rsp).reshape(*(list(p.shape) + [self.x] * newdim))
    
    @check_cache
    def prob_topk(self, k : int):
        r"""
        Calculate the joint probability distribution of top-k symbol the Markov Chain.
        - P(X_1, X_2, ..., X_k)

        Parameters:
            k (int): the index ( 1 begin ) of the Markov Chain sequence.
        
        Returns:
            array_like: the joint probability distribution of top-k symbol the Markov Chain.
            shape = (x, x, ..., x) ( k times )
        """
        if k < 0:
            raise ValueError("`k` must be greater than or equal to 0.")
        if k == 0:
            warnings.warn("`0` index will be ignored. The probability distribution of X_0 is 1.")
        if k == self.m:
            p_k = self.P_base_m
        elif k < self.m:
            p = self.prob_topk(k + 1)
            p_k = np.sum(p, axis=-1)
        else: 
            p = self.prob_topk(k - 1)
            p_k = self._prob_step(p, 1, 1)
        return p_k
    
    @deprecate
    def prob_k_old(self, k):
        r"""
        Calculate the probability distribution of k-th or joint probability distribution of X_k symbol the Markov Chain. For example, if k=2, return the probability distribution of X_2, if k=[1, 3], return the joint probability distribution of X_1,X_3.
        - Deprecated, recommend using `prob_k` instead. 2025-05-03
        - k = list(range(1, k+1)) -> P(X_1, X_2, ..., X_k) equivalent to `self.prob_topk(k)`
        - k = [1, 3] -> P(X_1, X_3)
        - k = [1, 2, 3] -> P(X_1, X_2, X_3)

        Parameters:
            k (int or int_list ): the index ( 1 begin ) of the Markov Chain sequence.
        
        Returns:
            array_like: the probability distribution.
            shape = (x, x, ..., x) ( len(k) times )
        """
        if isinstance(k, int):
            k = [k]
        if not np.all( map(lambda x: isinstance(x, int), k) ):
            raise ValueError("`k` must be int or int_list.")
        if min(k) < 0:
            raise ValueError("`k` or the elem of `k` must be postive.")
        maxk = max(k)
        sum_axis = [i for i in range(maxk) if i+1 not in k]
        return self.prob_topk(maxk).sum(axis=tuple(sum_axis))
    

    def prob_k(self, k):
        r"""
        Calculate the probability distribution of k-th or joint probability distribution of X_k symbol the Markov Chain. For example, if k=2, return the probability distribution of X_2, if k=[1, 3], return the joint probability distribution of X_1,X_3.
        - k = list(range(1, k+1)) -> P(X_1, X_2, ..., X_k) equivalent to `self.prob_topk(k)`
        - k = [1, 3] -> P(X_1, X_3)
        - k = [1, 2, 3] -> P(X_1, X_2, X_3)
        
        Parameters:
            k (int or int_list ): the index ( 1 begin ) of the Markov Chain sequence.
        
        Returns:
            array_like: the probability distribution.
            shape = (x, x, ..., x) ( len(k) times )
        """
        if isinstance(k, int):
            k = [k]
        assert len(k) > 0, "k(list) can't be empty."
        assert np.all( map(lambda x: isinstance(x, int) and x >= 0, k) ), "k must be int or int_list and k >= 0."
        assert np.all( np.diff(k) > 0 ),  "the elem of k must be increasing strictly."
        p = self.prob_topk(self.m)
        k_now = list(range(1, self.m+1))
        k_forward = k_now + [i for i in k if i > self.m]
        for i in self._get_steps(k_forward):
            step = i - k_now[-1]
            newdim = min(self.m, step)
            p = self._prob_step(p, step, newdim)
            k_now = k_now + list(range(i-newdim+1, i+1))
        axis_sum = [i for i in range(len(k_now)) if k_now[i] not in k]
        p = p.sum(axis=tuple(axis_sum))
        return p
    

if __name__ == "__main__":
    pass
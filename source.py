from utils import * 
from chain import MarkovChain, MemoryLessChain
import numpy as np


class Source():
    r"""
        A class to represent a discrete source.
        This class works not by itself, but by creating a new Class inherited from `MarkovChain` or `MemoryLessChain`.
        In it's `__new__` method, it creates a new class (`MemoryLessSource` or `MarkovSource` depended on parma `source_type`) with the same attributes (without `__new__` method) as the `Source` class.
        Use `Source` with default parameters to create a 2-symbol MemoryLessSource .

        Parameters:
            P_trans (array_like): m-order transition probability matrix for MarkovSource; probablity distribution for MemoryLessSource ( elem_num = x**(m+1) ) 
            x (int, optional, default=2): Number of symbols.
            m (int, optional, default=0): Order of the Markov chain. For MemoryLessSource, m = 0.
            source_type (str, optional, default='MemoryLess'): Type of the source. Can be 'MemoryLess' or 'Markov'.
    """
    def __new__(cls, P_trans, x=2, m=0, source_type='MemoryLess'):
        attr = dict(cls.__dict__)
        del attr['__new__']
        if source_type == 'MemoryLess':
            new_cls = type('MemoryLessSource', (MemoryLessChain,), attr)
        elif source_type == 'Markov':
            new_cls = type('MarkovSource', (MarkovChain,), attr)
        else:
            raise ValueError("Invalid source_type. Must be 'MemoryLess' or 'MarkovChain'.")
        instance = new_cls(P_trans, x, m)
        return instance

    def __init__(self, P_trans, x, m):
        super(self.__class__, self).__init__(P_trans, x, m)
        print(f"ChildClass initialized with parameter")

    # ! TODO
    def random_sequence(self, k, n, symbol=None, sep = '', prior=None, placeholder=None):
        r"""
        Generate random output sequences composed of symbols in source.

        Parameters:
            k (int): the length of the sequence.
            n (int): the number of sequence.
            symbol (str_list, dependedly optional): the symbol to generate. If None, generate symbol from ['0', '1', ..., '9', 'A', ... 'Z'] (36 totally). if the number of symbols greater than 36, `symbol` is needed.
            sep (str, optional): the separator between symbols. Default is ''. It is recommended to use when you have different length of symbols or the output sequence can't be decoded uniquely.
            prior (str, optional): the prior symbol of the chain. A random string composed of symbols in `symbol` under the probability distribution of the sequence. if used, keep consistent with `sep`. if the prior symbol is not adjacent in index, use `placeholder` to fill the gap. 
            placeholder (str, optional): the placeholder to fill the gap in `prior`. Default is None. Remember to use an ASCII char that is independent on `symbol` and `sep`, `@`, `*` or `#` may be good choices.

        Returns:
            str_list: a list of the generated sequence with `n` str.
        """
        assert self.x > 36 and symbol is not None, "If the number of symbols is greater than 36, `symbol` is needed."
        if symbol is None:
            symbol = [str(i) for i in range(min(self.x, 10))]
            symbol += [chr(i) for i in range(65, 65 + self.x - 10)] if self.x > 10 else []
        else :
            assert len(symbol) == self.x, "The length of `symbol` must be equal to the number of symbols."
        


if __name__ == "__main__":
    pass
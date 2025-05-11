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
            symbol (str_list, dependedly optional): the symbol of the source. If None, `symbol` generated from ['0', '1', ..., '9', 'A', ... 'Z'] (36 totally). if the number of symbols greater than 36, `symbol` is needed.
    """
    def __new__(cls, P_trans, x=2, m=0, symbol=None ,source_type='MemoryLess'):
        attr = dict(cls.__dict__)
        del attr['__new__']
        if source_type == 'MemoryLess':
            new_cls = type('MemoryLessSource', (MemoryLessChain,), attr)
        elif source_type == 'Markov':
            new_cls = type('MarkovSource', (MarkovChain,), attr)
        else:
            raise ValueError("Invalid source_type. Must be 'MemoryLess' or 'MarkovChain'.")
        instance = new_cls(P_trans, x, m, symbol)
        return instance


    def __init__(self, P_trans, x, m, symbol):
        super(self.__class__, self).__init__(P_trans, x, m)
        assert symbol is not None if self.x > 36 else True, "If the number of symbols is greater than 36, `symbol` is needed."
        if symbol is None:
            symbol = [str(i) for i in range(min(self.x, 10))]
            symbol += [chr(i) for i in range(ord('A'), ord('A') + self.x - 10)] if self.x > 10 else []
        else :
            assert len(symbol) == self.x, "The length of `symbol` must be equal to the number of symbols."
        self.symbol = symbol


    @check_cache
    def _ten2any(self, n, base, _len):
        digits = []
        while n:
            digits.append(n % base)
            n //= base
        if _len is not None:
            digits += [0] * (_len - len(digits))
        return digits[::-1]

    def random_sequence_gen(self, k, n=None, sep = '', prior=None, placeholder=None):
        r"""
        Generate random an iterator of output sequences composed of symbols in source.

        Parameters:
            k (int): the length of the sequence.
            n (int, optional): the number of sequence. if None, return an iterator `func` with required args `n`, else return an iterator with `n` iters. 
            sep (str, optional): the separator between symbols. Default is ''. It is recommended to use when you have different length of symbols or the output sequence can't be decoded uniquely.
            prior (str, optional): the prior symbol of the chain. A random string composed of symbols in `symbol` under the probability distribution of the sequence. if used, keep consistent with `sep`. if the prior symbol is not adjacent in index, use `placeholder` to fill the gap. 
            placeholder (str, optional): the placeholder to fill the gap in `prior`. Default is None. Remember to use an ASCII char that is independent on `symbol` and `sep`, `@`, `*` or `#` may be good choices.

        Returns:
            iterator: an iterator or iterator func that generates random sequences of length `k`.
        
        Example:
            ```python
            s = Source([0.75, 0.25], symbol=['0', '1'])
            itera = s.random_sequence_gen(9, prior='0-1-@-0-1-@-0', placeholder='@', sep='-')
            for i in itera(5):
                print(i)
            ```
        """
        
        prior_list = list(prior) if sep == '' else prior.split(sep)
        prior_indexs = [ i+1 for i, s in enumerate(prior_list) if s != placeholder ]
        prior_symbols = list(filter(lambda x: x != placeholder, prior_list))
        cal_indexs = [i for i in range(1, k+1) if i not in prior_indexs]
        p = self.prob_k(cal_indexs).flatten()
        template = sep.join([prior_symbols.pop(0) if i+1 in prior_indexs else '{}' for i in range(k)])
        
        def _sequence_iterate(num):
            for i in range(num):
                index_ten = np.random.choice(self.x**len(cal_indexs), p=p, replace=True)
                index_list = self._ten2any(index_ten, self.x, len(cal_indexs))
                yield template.format(*list(map(lambda x: self.symbol[x], index_list)))

        return _sequence_iterate(n) if n is not None else _sequence_iterate

    def symbol_gen():
        pass

if __name__ == "__main__":
    pass

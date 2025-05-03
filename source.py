from utils import * 
from markov import MarkovChain
import numpy as np

# ! TODO
class MarkovSource(MarkovChain):
    def __init__(self, P_trans, m, x):
        super().__init__(P_trans, m, x)


    def generate_note(self, k, n, note=None, sep = '', prior=None):
        r"""
        Generate notes of the Markov Chain.

        Parameters:
            k (int): the length of the note.
            n (int): the number of notes to generate.
            note (str_list, optional): the note to generate. If None, generate note from ['0', '1', ..., '9', 'A', ... 'Z'] (36 totally). if the number of symbols greater than 36, `note` is needed.
            sep (str, optional): the separator between notes. Default is ''. It is recommended to use when you have different length of symbols or the output sequence can't be decoded uniquely.
            prior (str, optional): the prior note of the chain. A random string composed of symbols in `note` under the probability distribution of the Markov Chain. if used, keep consistent with `sep`.
        Returns:
            str_list: a list the generated notes with `n` str.

        example:
            mrkv = MarkovChain(P_trans, 1, 3)
            mrkv.setup(...)
            print(mrkv.generate_note(3, 5, sep='-', prior='0-1'))
            
        """
        assert self.x > 36 and note is not None, "If the number of symbols is greater than 36, `note` is needed."
        if note is None:
            note = [str(i) for i in range(min(self.x, 10))]
            note += [chr(i) for i in range(65, 65 + self.x - 10)] if self.x > 10 else []
        for i in range(n):
            pass

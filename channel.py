from utils import * 
import chain
from chain import MarkovChain, MemoryLessChain
import numpy as np


# ! TODO
class MarkovChannel(MarkovChain):
    r"""
    A class to represent a Markov Channel.
    """
    def __init__(self, P_trans, m, x):
        super().__init__(P_trans, m, x)
        pass

    def channel_pass():
        pass
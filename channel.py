from chain import MarkovChain
from utils import __float_dtype__, _prob_dstrbt_check
from probability import mean_mutual_information
import numpy as np


class Channel:
    r"""
    Unified channel factory.

    Similar to `Source`, this class dynamically creates and returns a
    concrete channel instance:
    - `MemoryLessChannel` when `channel_type='MemoryLess'`
    - `MarkovChannel` when `channel_type='Markov'`

    Parameters:
        P_trans (array_like): Channel conditional matrix.
        x (int, optional): Alphabet size of input and output. Default is 2.
        m (int, optional): Channel memory order, used for Markov channel.
        input_symbol (list[str], optional): Input symbol table.
        output_symbol (list[str], optional): Output symbol table.
        channel_type (str, optional): 'MemoryLess' or 'Markov'.
    """

    def __new__(
        cls,
        P_trans,
        x=2,
        m=1,
        input_symbol=None,
        output_symbol=None,
        channel_type='MemoryLess',
    ):
        attr = dict(cls.__dict__)
        attr.pop('__new__', None)

        if channel_type == 'MemoryLess':
            new_cls = type('MemoryLessWrappedChannel', (MemoryLessChannel,), attr)
            return new_cls(
                P_trans=P_trans,
                x=x,
                input_symbol=input_symbol,
                output_symbol=output_symbol,
            )
        if channel_type == 'Markov':
            new_cls = type('MarkovWrappedChannel', (MarkovChannel,), attr)
            return new_cls(
                P_trans=P_trans,
                x=x,
                m=m,
                input_symbol=input_symbol,
                output_symbol=output_symbol,
            )
        raise ValueError("Invalid `channel_type`. Must be 'MemoryLess' or 'Markov'.")


class MarkovChannel(MarkovChain):
    r"""
    Discrete Markov channel.

    The channel is parameterized by a conditional matrix:
        P(y_t | x_{t-m+1}, ..., x_t)

    where `m` is the channel memory order over input symbols.

    Parameters:
        P_trans (array_like): Conditional transition matrix with shape (x**m, x).
        x (int): Alphabet size of both input and output symbols.
        m (int): Channel memory order. `m=1` is a memoryless channel.
        input_symbol (list[str], optional): Input symbol table.
        output_symbol (list[str], optional): Output symbol table.
    """

    def __init__(self, P_trans, x=2, m=1, input_symbol=None, output_symbol=None):
        super().__init__(P_trans, x, m)
        self.input_symbol = self._build_symbol_table(x, input_symbol)
        self.output_symbol = self._build_symbol_table(x, output_symbol)

    @staticmethod
    def _build_symbol_table(x, symbol):
        if symbol is None:
            if x > 36:
                raise ValueError("If `x > 36`, `symbol` must be provided.")
            symbol = [str(i) for i in range(min(x, 10))]
            if x > 10:
                symbol += [chr(i) for i in range(ord('A'), ord('A') + x - 10)]
        if len(symbol) != x:
            raise ValueError("The length of `symbol` must be equal to `x`.")
        return list(symbol)

    @staticmethod
    def _normalize_prob(p):
        p = np.asarray(p, dtype=__float_dtype__).reshape(-1)
        s = np.sum(p)
        if np.isclose(s, 0.0):
            raise ValueError("Probability vector sums to 0.")
        return p / s

    def _to_index_seq(self, seq, symbol_table, sep=''):
        if seq is None:
            return []
        if isinstance(seq, np.ndarray):
            seq = seq.tolist()
        if isinstance(seq, str):
            seq = list(seq) if sep == '' else seq.split(sep)

        idx_list = []
        for item in seq:
            if isinstance(item, (int, np.integer)):
                idx = int(item)
                if idx < 0 or idx >= self.x:
                    raise ValueError(f"Index `{idx}` out of range [0, {self.x - 1}].")
                idx_list.append(idx)
            else:
                try:
                    idx_list.append(symbol_table.index(item))
                except ValueError as exc:
                    raise ValueError(f"Unknown symbol `{item}`.") from exc
        return idx_list

    def _index_to_seq(self, idx_list, symbol_table, sep=''):
        out = [symbol_table[i] for i in idx_list]
        return ''.join(out) if sep == '' else sep.join(out)

    def _contexts(self, x_idx, prior_idx):
        x_idx = list(x_idx)
        if self.m == 1:
            return [[x] for x in x_idx]

        need = self.m - 1
        if len(prior_idx) != need:
            raise ValueError(f"`prior` must contain exactly {need} symbols when m={self.m}.")

        history = list(prior_idx) + x_idx
        return [history[i:i + self.m] for i in range(len(x_idx))]

    def channel_pass(self, x_seq, sep='', prior=None, rng=None, return_index=False):
        r"""
        Pass an input sequence through the channel and sample output symbols.

        Parameters:
            x_seq (sequence or str): Input symbol sequence.
            sep (str, optional): Symbol separator when string sequence is used.
            prior (sequence or str, optional): Required when m > 1, length must be m-1.
            rng (np.random.Generator, optional): Random generator.
            return_index (bool, optional): If True, return index list.

        Returns:
            Same symbolic format as input when `return_index=False`, otherwise index list.
        """
        x_idx = self._to_index_seq(x_seq, self.input_symbol, sep=sep)
        prior_idx = self._to_index_seq(prior, self.input_symbol, sep=sep) if prior is not None else []

        contexts = self._contexts(x_idx, prior_idx)
        rng = np.random.default_rng() if rng is None else rng

        y_idx = []
        for ctx in contexts:
            p = self._normalize_prob(self._single_step_prob(ctx))
            y_idx.append(int(rng.choice(self.x, p=p, replace=True)))

        if return_index:
            return y_idx
        return self._index_to_seq(y_idx, self.output_symbol, sep=sep)

    def channel_prob(self, x_seq, y_seq, sep='', prior=None):
        r"""
        Compute P(y_seq | x_seq) under the channel model.
        """
        x_idx = self._to_index_seq(x_seq, self.input_symbol, sep=sep)
        y_idx = self._to_index_seq(y_seq, self.output_symbol, sep=sep)
        if len(x_idx) != len(y_idx):
            raise ValueError("`x_seq` and `y_seq` must have the same length.")

        prior_idx = self._to_index_seq(prior, self.input_symbol, sep=sep) if prior is not None else []
        contexts = self._contexts(x_idx, prior_idx)

        prob = 1.0
        for ctx, y in zip(contexts, y_idx):
            p = self._normalize_prob(self._single_step_prob(ctx))
            prob *= float(p[y])
        return prob


class MemoryLessChannel(MarkovChannel):
    r"""
    Discrete memoryless channel (DMC), i.e. Markov channel with m=1.

    Parameters:
        P_trans (array_like): Channel matrix P(y|x), shape (x, x).
        x (int): Alphabet size.
        input_symbol (list[str], optional): Input symbol table.
        output_symbol (list[str], optional): Output symbol table.
    """

    def __init__(self, P_trans, x=2, input_symbol=None, output_symbol=None):
        super().__init__(P_trans=P_trans, x=x, m=1, input_symbol=input_symbol, output_symbol=output_symbol)

    @property
    def P_channel(self):
        return self.P_trans.reshape(self.x, self.x)

    def output_prob(self, pX):
        r"""
        Compute output distribution p(Y) from input distribution p(X).
        """
        pX = np.asarray(pX, dtype=__float_dtype__).reshape(self.x)
        if not _prob_dstrbt_check(pX):
            raise ValueError("`pX` must be a valid probability distribution.")
        return np.dot(pX, self.P_channel)

    def joint_prob(self, pX):
        r"""
        Compute joint distribution p(X,Y) from p(X) and channel matrix P(Y|X).
        """
        pX = np.asarray(pX, dtype=__float_dtype__).reshape(self.x)
        if not _prob_dstrbt_check(pX):
            raise ValueError("`pX` must be a valid probability distribution.")
        return pX.reshape(self.x, 1) * self.P_channel

    def mean_mutual_information(self, pX):
        r"""
        Compute I(X;Y) for a memoryless channel given input prior p(X).
        """
        pX = np.asarray(pX, dtype=__float_dtype__).reshape(self.x)
        if not _prob_dstrbt_check(pX):
            raise ValueError("`pX` must be a valid probability distribution.")

        pY = self.output_prob(pX)
        pXY = self.joint_prob(pX).reshape(-1)
        return mean_mutual_information(pX, pY, pXY)


if __name__ == "__main__":
    # quick self-check
    dmc = Channel(
        [0.9, 0.1, 0.1, 0.9],
        x=2,
        input_symbol=['0', '1'],
        output_symbol=['0', '1'],
        channel_type='MemoryLess',
    )
    x = '010011'
    y = dmc.channel_pass(x)
    print(x, '->', y)
    print('P(y|x)=', dmc.channel_prob(x, y))

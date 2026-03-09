import heapq
import itertools
from collections import Counter
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


class HuffmanNode:
    """Node in a Huffman tree."""

    def __init__(self, symbol: Any = None, weight: float = 0.0, left=None, right=None):
        self.symbol = symbol
        self.weight = float(weight)
        self.left = left
        self.right = right

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def _as_symbol_list(data, sep: str = ''):
    """Convert supported input formats to a symbol list."""
    if data is None:
        raise ValueError("`data` cannot be None.")

    if isinstance(data, np.ndarray):
        data = data.tolist()

    if isinstance(data, str):
        symbols = list(data) if sep == '' else data.split(sep)
    else:
        try:
            symbols = list(data)
        except TypeError:
            symbols = [data]

    if len(symbols) == 0:
        raise ValueError("`data` cannot be empty.")
    return symbols


def _weights_from_probabilities(probabilities, symbols=None) -> Dict[Any, float]:
    if isinstance(probabilities, dict):
        items = probabilities.items()
    else:
        probs = np.asarray(probabilities, dtype=float).reshape(-1)
        if symbols is None:
            symbols = list(range(len(probs)))
        if len(symbols) != len(probs):
            raise ValueError("`symbols` length must match `probabilities` length.")
        items = zip(symbols, probs)

    weights = {}
    for symbol, weight in items:
        w = float(weight)
        if w < 0:
            raise ValueError("Probability/weight must be non-negative.")
        if w > 0:
            weights[symbol] = w

    if not weights:
        raise ValueError("At least one symbol must have positive probability/weight.")
    return weights


def build_symbol_distribution(data, sep: str = '') -> Dict[Any, float]:
    """
    Build empirical probability distribution from real symbol sequence data.

    Parameters:
        data: sequence-like symbols or str.
        sep (str): split separator when `data` is string and symbol length > 1.

    Returns:
        dict: {symbol: probability}
    """
    symbols = _as_symbol_list(data, sep=sep)
    counts = Counter(symbols)
    total = float(sum(counts.values()))
    return {k: v / total for k, v in counts.items()}


def build_huffman_tree(probabilities, symbols=None) -> HuffmanNode:
    """
    Build Huffman tree from probabilities/weights.

    Parameters:
        probabilities: list/array-like, or dict {symbol: prob}.
        symbols: optional symbol list when `probabilities` is an array-like.

    Returns:
        HuffmanNode: root node of the Huffman tree.
    """
    weights = _weights_from_probabilities(probabilities, symbols=symbols)

    order = itertools.count()
    heap = []
    for symbol, weight in weights.items():
        node = HuffmanNode(symbol=symbol, weight=weight)
        heapq.heappush(heap, (weight, next(order), node))

    while len(heap) > 1:
        w1, _, n1 = heapq.heappop(heap)
        w2, _, n2 = heapq.heappop(heap)
        merged = HuffmanNode(symbol=None, weight=w1 + w2, left=n1, right=n2)
        heapq.heappush(heap, (merged.weight, next(order), merged))

    return heap[0][2]


def generate_huffman_codes(root: HuffmanNode) -> Dict[Any, str]:
    """Generate Huffman code table from a Huffman tree root."""
    if root is None:
        raise ValueError("`root` cannot be None.")

    codes = {}

    if root.is_leaf:
        # Single-symbol edge case.
        codes[root.symbol] = '0'
        return codes

    def traverse(node: HuffmanNode, code: str):
        if node.is_leaf:
            codes[node.symbol] = code
            return
        traverse(node.left, code + '0')
        traverse(node.right, code + '1')

    traverse(root, '')
    return codes


def _build_decode_tree(codes: Dict[Any, str]):
    root = {}
    for symbol, code in codes.items():
        if code == '':
            raise ValueError("Empty code detected. Use at least one bit per symbol.")

        node = root
        for bit in code:
            if bit not in ('0', '1'):
                raise ValueError(f"Invalid bit `{bit}` in code for symbol `{symbol}`.")
            node = node.setdefault(bit, {})

        if '_symbol' in node:
            raise ValueError("Duplicated Huffman code detected.")
        node['_symbol'] = symbol
    return root


class HuffmanCodec:
    """
    Huffman encoder/decoder for symbolic data.

    Use `from_data(...)` when you have real symbol samples,
    or `from_probabilities(...)` when you have a known distribution.
    """

    def __init__(self, codes: Dict[Any, str], probabilities: Optional[Dict[Any, float]] = None):
        if not codes:
            raise ValueError("`codes` cannot be empty.")
        self.codes = dict(codes)
        self.decode_tree = _build_decode_tree(self.codes)
        self.probabilities = dict(probabilities) if probabilities is not None else None

    @classmethod
    def from_probabilities(cls, probabilities, symbols=None):
        weights = _weights_from_probabilities(probabilities, symbols=symbols)
        total = float(sum(weights.values()))
        probs = {k: v / total for k, v in weights.items()}
        root = build_huffman_tree(probs)
        codes = generate_huffman_codes(root)
        return cls(codes, probabilities=probs)

    @classmethod
    def from_data(cls, data, sep: str = ''):
        probs = build_symbol_distribution(data, sep=sep)
        root = build_huffman_tree(probs)
        codes = generate_huffman_codes(root)
        return cls(codes, probabilities=probs)

    def encode(self, data, sep: str = '') -> str:
        symbols = _as_symbol_list(data, sep=sep)
        try:
            return ''.join(self.codes[s] for s in symbols)
        except KeyError as exc:
            raise KeyError(f"Unknown symbol in data: {exc.args[0]}") from exc

    def decode(self, bitstream, sep: str = '', return_list: bool = False):
        if isinstance(bitstream, (list, tuple, np.ndarray)):
            bitstream = ''.join(str(int(b)) for b in bitstream)
        if not isinstance(bitstream, str):
            raise TypeError("`bitstream` must be a str or iterable of bits.")

        out = []
        node = self.decode_tree
        for bit in bitstream:
            if bit not in ('0', '1'):
                raise ValueError(f"Invalid bit `{bit}` in bitstream.")
            if bit not in node:
                raise ValueError("Invalid bitstream for current Huffman table.")
            node = node[bit]

            if '_symbol' in node:
                out.append(node['_symbol'])
                node = self.decode_tree

        if node is not self.decode_tree:
            raise ValueError("Bitstream ended in the middle of a codeword.")

        if return_list:
            return out

        if all(isinstance(s, str) for s in out):
            return ''.join(out) if sep == '' else sep.join(out)
        return out

    def average_code_length(self) -> Optional[float]:
        if self.probabilities is None:
            return None
        return float(sum(self.probabilities[s] * len(self.codes[s]) for s in self.probabilities))


# Backward-compatible helper.
def huffman_encoding(probabilities, symbols=None):
    """
    Build Huffman code table from probabilities.

    Backward compatible with old `huffman_encoding(probabilities)` usage.
    """
    return HuffmanCodec.from_probabilities(probabilities, symbols=symbols).codes


def huffman_encode(data, probabilities=None, symbols=None, codes=None, sep: str = '') -> Tuple[str, Dict[Any, str]]:
    """
    Convenience encoding function.

    Modes:
    - pass `codes` directly
    - pass `probabilities` (+ optional `symbols`)
    - or fit from `data` empirical distribution
    """
    if codes is not None:
        codec = HuffmanCodec(codes)
    elif probabilities is not None:
        codec = HuffmanCodec.from_probabilities(probabilities, symbols=symbols)
    else:
        codec = HuffmanCodec.from_data(data, sep=sep)
    return codec.encode(data, sep=sep), codec.codes


def huffman_decode(bitstream, codes, sep: str = '', return_list: bool = False):
    """Convenience decoding function from code table."""
    codec = HuffmanCodec(codes)
    return codec.decode(bitstream, sep=sep, return_list=return_list)


if __name__ == "__main__":
    # quick self-check with real symbol data
    sample = "0-1-0-0-1-1-0-0-1"
    codec = HuffmanCodec.from_data(sample, sep='-')
    bits = codec.encode(sample, sep='-')
    restored = codec.decode(bits, sep='-')

    print("codes:", codec.codes)
    print("bits:", bits)
    print("restored:", restored)

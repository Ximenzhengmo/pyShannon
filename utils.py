import numpy as np
import functools
import warnings
from _thread import RLock

__float_dtype__ = np.float32  # Default float type for numpy arrays
__check_cache_size__ = 16

def check_cache(func):
    cache_dict = {}
    lock = RLock()
    @functools.wraps(func)
    def _check_cache(*args, **kwargs):
        key_parts = [id(arg) if isinstance(arg, np.ndarray) else arg for arg in args] \
        + [(k, id(v)) if isinstance(v, np.ndarray) else (k, v) for k, v in kwargs.items()]
        key = tuple(key_parts)
        with lock:
            if key in cache_dict:
                return cache_dict[key]
        result = func(*args, **kwargs)
        with lock:
            cache_dict[key] = result
        return result
    
    def clear_cache():
        nonlocal cache_dict
        with lock:
            if len(cache_dict) > __check_cache_size__:
                cache_dict = {} 
    
    _check_cache.clear = clear_cache
    return _check_cache


def deprecate(func):
    @functools.wraps(func)
    def _deprecate(*args, **kwargs):
        warnings.warn(f"{func.__name__} is deprecated and will be removed in future versions.", DeprecationWarning)
        return func(*args, **kwargs)
    return _deprecate


@check_cache
def _prob_scale_check(p):
    """
    Check if the probabilities are in the range [0, 1].
    """
    if 0. <= p.all() <= 1. :
        return True
    return False

@check_cache
def _prob_sum_one_check(p, axis=None):
    """
    Check if the sum of the probabilities is equal to 1.
    """
    if np.all(np.abs(np.sum(p, axis=axis) - 1) > np.finfo(p.dtype).eps):
        return False
    return True

@check_cache
def _prob_dstrbt_check(p, axis=None):
    return _prob_scale_check(p) and _prob_sum_one_check(p, axis=axis)

def self_information(p):
    r"""
    Calculate the self-information of an event with probability p.
    - p(x) -> I(x) = -log2(p(x))
    - p(x|y) -> I(x|y) = -log2(p(x|y))
    
    Parameters:
        p (array_like): Probability of the event.

    Returns:
        array_like: Self-information of the event.
    """
    try:
        p = np.asarray(p, dtype=__float_dtype__)  # Ensure p is a numpy array of floats
    except ValueError:
        raise ValueError("Input must be a number or an array-like of numbers.")
        
    assert _prob_scale_check(p), "Probabilities `p` must be in the range [0, 1]."

    p = np.where(p != 0, p, np.finfo(p.dtype).eps)  # Add a small epsilon to avoid log(0)
    return -np.log2(p)


def mutual_information(px, px_y):
    r"""
    Calculate the mutual-information of an event with probability p.
    - p(x), p(x|y) -> I(x; y) = -log2(p(x)) - log2(p(x|y))
    - p(x|z), p(x|y,z) -> I(x; y|z) = -log2(p(x|z)) - log2(p(x|y,z))

    Parameters:
        px   (array_like): Probability of the event x - p(x).
        px_y (array_like): Probability of the event x given y  p(x|y).

    Returns:
        array_like: Self-information of the event.
    """
    return self_information(px) - self_information(px_y)


def Entropy(p, pXY=None):
    r"""
    Calculate the entropy of a probability distribution.
    - entropy     :  p(X) -> H(X) = -sum(p(X) * log2(p(X)))
    - joint-entropy   : p(XY) -> H(XY) = -sum(p(XY) * log2(p(XY)))
    - conditional-entropy : p(X|Y), pXY=p(XY) -> H(X|Y) = -sum(p(XY) * log2(p(X|Y)))
        
    Parameters:
        p (array_like): Probability distribution.
        pXY (array_like, optional): joint-prob p(XY), used for assigning weight for I(x|y) (for conditional-entropy or joint-entropy). If None, uses p as pXY (for entropy or joint-entropy).

    Returns:
        float: Entropy of the distribution H(X) or H(XY) or H(X|Y).
    """

    try:
        p = np.asarray(p, dtype=__float_dtype__)  # Ensure p is a numpy array of floats
    except ValueError:
        raise ValueError("Input must be a number or an array-like of numbers.")

    if pXY is None:
        pXY = p
    else:
        try:
            pXY = np.asarray(pXY, dtype=__float_dtype__)  # Ensure weight is a numpy array of floats
        except ValueError:
            raise ValueError("Input must be a number or an array-like of numbers.")

    assert _prob_dstrbt_check(pXY), "Probabilities `pXY` must be in the range [0, 1] and sum(pXY) = 1."

    return -np.dot( self_information(p),  pXY )


def mean_contitional_mutial_information(pX, pX_y):
    r"""
    Calculate the mean conditional mutual information of a probability distribution.
    - pX, pX_y -> I(X; y) = sum_X( p(X_i) * I(X_i; y) )

    Parameters:
        pX   (array_like): Probability of the event X - p(X).
        pX_y (array_like): Probability of the event X given single event y  p(X|y).

    Returns:
        float: Mean conditional mutual information of the distribution I(X; y).
    """
    try:
        pX = np.asarray(pX, dtype=__float_dtype__)  # Ensure p is a numpy array of floats
        pX_y = np.asarray(pX_y, dtype=__float_dtype__)  # Ensure p is a numpy array of floats
    except ValueError:
        raise ValueError("Input must be a number or an array-like of numbers.")
    
    assert _prob_dstrbt_check(pX), "Probabilities `pX` must be in the range [0, 1]. and sum(pX) = 1."

    return np.dot(pX, mutual_information(pX, pX_y)) 


def mean_mutual_information(pX, pY, pXY):
    r"""
    Calculate the mean mutual information of a probability distribution.
    - pX, pY, pXY -> I(X; Y) = sum_X( p(X_i) * I(X_i; Y) )

    Parameters:
        pX   (array_like): Probability of the event X - p(X).
        pY   (array_like): Probability of the event Y - p(Y).
        pXY  (array_like): Probability of the event XY - p(XY).

    Returns:
        float: Mean mutual information of the distribution I(X; Y).
    """
    try:
        pX = np.asarray(pX, dtype=__float_dtype__)  # Ensure p is a numpy array of floats
        pY = np.asarray(pY, dtype=__float_dtype__)  # Ensure p is a numpy array of floats
        pXY = np.asarray(pXY, dtype=__float_dtype__)  # Ensure weight is a numpy array of floats
    except ValueError:
        raise ValueError("Input must be a number or an array-like of numbers.")
    return Entropy(pX) + Entropy(pY) - Entropy(pXY)


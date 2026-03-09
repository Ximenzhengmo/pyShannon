from checker import ProbabilityChecker, ScaleoneChecker, AsarrayChecker
import numpy as np

probChecker_lastaxis = ProbabilityChecker(axis=-1)
probChecker_Allaxis = ProbabilityChecker(axis=None)
scaleoneChecker = ScaleoneChecker()
asarrayChecker = AsarrayChecker()

def _self_information_impl(p):
    p = np.where(p != 0, p, np.finfo(p.dtype).eps)  # Add a small epsilon to avoid log(0)
    return -np.log2(p)


def _entropy_impl(p):
    return np.dot(_self_information_impl(p), p)


def _mutual_information_impl(px, px_y):
    return _self_information_impl(px) - _self_information_impl(px_y)


def _check_asarray(value, param_name):
    return asarrayChecker.arg_checker(value, param_name=param_name)


def _check_scaleone(value, param_name):
    return scaleoneChecker.arg_checker(value, param_name=param_name)


def _check_probability(value, param_name):
    return probChecker_Allaxis.arg_checker(value, param_name=param_name)


def self_information(p):
    r"""
    Calculate the self-information of an event with probability p.
    - p(x) -> I(x) = -log2(p(x))
    - p(x|y) -> I(x|y) = -log2(p(x|y))
    
    Parameters:
        p (array_like vector): Probability of the event.

    Returns:
        array_like: Self-information of the event.
    """
    p = _check_scaleone(p, 'p')
    return _self_information_impl(p)

def mutual_information(px, px_y):
    r"""
    Calculate the mutual-information of an event with probability p.
    - p(x), p(x|y) -> I(x; y) = -log2(p(x)) - log2(p(x|y))
    - p(x|z), p(x|y,z) -> I(x; y|z) = -log2(p(x|z)) - log2(p(x|y,z))

    Parameters:
        px   (array_like vector): Probability of the event x - p(x).
        px_y (array_like vector): Probability of the event x given y  p(x|y).

    Returns:
        array_like: Self-information of the event.
    """
    px = _check_scaleone(px, 'px')
    px_y = _check_scaleone(px_y, 'px_y')
    return _mutual_information_impl(px, px_y)

def Entropy(p):
    r"""
    Calculate the entropy of a probability distribution.
    - entropy     :  p(X) -> H(X) = -sum(p(X) * log2(p(X)))
    - joint-entropy   : p(XY) -> H(XY) = -sum(p(XY) * log2(p(XY)))
        
    Parameters:
        p (array_like vector): Probability distribution.

    Returns:
        float: Entropy of the distribution H(X) or H(XY).
    """
    p = _check_probability(p, 'p')
    return _entropy_impl(p)

def conditional_entropy(p, pXY):
    r"""
    Calculate the conditional entropy of a probability distribution.
    - conditional-entropy : p=p(X|Y), pXY=p(XY) -> H(X|Y) = -sum(p(XY) * log2(p(X|Y)))

    Parameters:
        p   (array_like, vector): Probability of the event X given Y - p(X|Y).
        pXY (array_like, vector): Joint probability of the event XY - p(XY).

    Returns:
        float: Conditional entropy of the distribution H(X|y).
    """
    p = _check_asarray(p, 'p')
    pXY = _check_probability(pXY, 'pXY')
    return np.dot(_self_information_impl(p), pXY)

def mean_contitional_mutial_information(pX, pX_y):
    r"""
    Calculate the mean conditional mutual information of a probability distribution.
    - pX, pX_y -> I(X; y) = sum_X( p(X_i) * I(X_i; y) )

    Parameters:
        pX   (array_like, vector): Probability of the event X - p(X).
        pX_y (array_like, vector): Probability of the event X given single event y  p(X|y).

    Returns:
        float: Mean conditional mutual information of the distribution I(X; y).
    """
    pX = _check_probability(pX, 'pX')
    pX_y = _check_asarray(pX_y, 'pX_y')
    return np.dot(pX, _mutual_information_impl(pX, pX_y))


def mean_mutual_information(pX, pY, pXY):
    r"""
    Calculate the mean mutual information of a probability distribution.
    - pX, pY, pXY -> I(X; Y) = sum_X( p(X_i) * I(X_i; Y) )

    Parameters:
        pX   (array_like, vector): Probability of the event X - p(X).
        pY   (array_like, vector): Probability of the event Y - p(Y).
        pXY  (array_like, vector): Probability of the event XY - p(XY).

    Returns:
        float: Mean mutual information of the distribution I(X; Y).
    """
    pX = _check_probability(pX, 'pX')
    pY = _check_probability(pY, 'pY')
    pXY = _check_probability(pXY, 'pXY')
    return _entropy_impl(pX) + _entropy_impl(pY) - _entropy_impl(pXY)


# Example usage
if __name__ == "__main__":
    pY = [0.5, 0.5]
    print(conditional_entropy(pY, pY))

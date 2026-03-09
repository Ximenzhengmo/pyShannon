from checker import inspector, ProbabilityChecker, ScaleoneChecker, AsarrayChecker
import numpy as np

probChecker_lastaxis = ProbabilityChecker(axis=-1)
probChecker_Allaxis = ProbabilityChecker(axis=None)
scaleoneChecker = ScaleoneChecker()
asarrayChecker = AsarrayChecker()

@inspector(args_to_check=True, checker=scaleoneChecker)
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
    p = np.where(p != 0, p, np.finfo(p.dtype).eps)  # Add a small epsilon to avoid log(0)
    return -np.log2(p)

@inspector(args_to_check=True, checker=scaleoneChecker)
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
    return self_information(px) - self_information(px_y)

@inspector(args_to_check=True, checker=probChecker_Allaxis)
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
    return np.dot( self_information(p),  p )

@inspector(args_to_check=('p'), checker=asarrayChecker)
@inspector(args_to_check=('pXY'), checker=probChecker_Allaxis)
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
    return np.dot( self_information(p),  pXY )

@inspector(args_to_check=('pX_y'), checker=asarrayChecker)
@inspector(args_to_check=('pX'), checker=probChecker_Allaxis)
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
    return np.dot(pX, mutual_information(pX, pX_y)) 


@inspector(args_to_check=True, checker=probChecker_Allaxis)
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
    return Entropy(pX) + Entropy(pY) - Entropy(pXY)


# Example usage
if __name__ == "__main__":
    pY = [0.5, 0.5]
    print(conditional_entropy(pY, pY))

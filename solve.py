import numpy as np
from probability import entropy

# Variable order: ['X', 'Y', 'Z']
# X states: ['0', '1']
# Y states: ['0', '1']
# Z states: ['0', '1']

tensor = np.zeros((2, 2, 2), dtype=np.float32)
tensor[0, 0, 0] = 0.3  # X=0, Y=0, Z=0
tensor[0, 1, 0] = 0.2  # X=0, Y=1, Z=0
tensor[1, 0, 0] = 0.4  # X=1, Y=0, Z=0
tensor[1, 1, 1] = 0.1  # X=1, Y=1, Z=1

def flatten(p):
    return np.asarray(p, dtype=np.float32).reshape(-1)

def reorder_marginal(tensor, subset):
    subset = tuple(subset)
    if not subset:
        return np.asarray(tensor, dtype=np.float32)
    keep_axes = tuple(axis for axis in range(tensor.ndim) if axis in subset)
    sum_axes = tuple(axis for axis in range(tensor.ndim) if axis not in subset)
    marginal = tensor.sum(axis=sum_axes) if sum_axes else tensor
    if keep_axes == subset:
        return np.asarray(marginal, dtype=np.float32)
    permutation = [keep_axes.index(axis) for axis in subset]
    return np.transpose(marginal, axes=permutation).astype(np.float32)

results = {}

# H(X)

p_x = reorder_marginal(tensor, (0,))
results['H(X)'] = entropy(flatten(p_x))

# H(Y)

p_y = reorder_marginal(tensor, (1,))
results['H(Y)'] = entropy(flatten(p_y))

# H(XY)

p_x_y = reorder_marginal(tensor, (0, 1))
results['H(XY)'] = entropy(flatten(p_x_y))

# I(X;Y)

results['I(X;Y)'] = results['H(X)'] + results['H(Y)'] - results['H(XY)']

# H(XZ)

p_x_z = reorder_marginal(tensor, (0, 2))
results['H(XZ)'] = entropy(flatten(p_x_z))

# H(XYZ)

p_x_y_z = reorder_marginal(tensor, (0, 1, 2))
results['H(XYZ)'] = entropy(flatten(p_x_y_z))

# I(XZ;Y)

results['I(XZ;Y)'] = results['H(XZ)'] + results['H(Y)'] - results['H(XYZ)']



# Print focused formulas

print("I(X;Y)", '=', results["I(X;Y)"])

print("I(XZ;Y)", '=', results["I(XZ;Y)"])
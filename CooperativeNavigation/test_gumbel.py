import numpy as np
def gumbel(size, eps=1e-10):
    """ Sample from Gumbel(0, 1)"""
    u = np.random.random(size)
    g = -np.log(-np.log(u + eps) + eps)
    return g


def gumbel_max_sample(x, is_prob=False):
    """ Draw a sample from P(X=k) prop x_k """
    if is_prob:
        x = np.log(x)

    g = gumbel(size=x.shape)
    return (g + x).argmax(axis=0)
def softmax(X, temperature = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    temperature (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """
    y = np.atleast_2d(X)

    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    y = y / float(temperature)
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    y = np.exp(y)

    # take the sum along the specified axis
    p = y / np.expand_dims(np.sum(y, axis = axis), axis)

    if len(X.shape) == 1:
        p = p.flatten()

    return p

def gumbel_softmax_sample(logits, temperature=1):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + gumbel(np.shape(logits))
    return softmax(y, temperature=temperature)

logits = [1, 4, 5, 1, 2]
classes = range(1, len(logits) + 1)
sample = gumbel_softmax_sample(logits, temperature=1)
print(sample)


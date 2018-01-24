"""A collection of utility functions used throughout Ankura"""

import numpy as np
import pickle
import functools
import os

try:
    import numba
    jit = functools.partial(numba.jit, nopython=True)
except ImportError:
    jit = lambda x:x


def random_projection(A, k):
    """Randomly projects the points (rows) of A into k-dimensions.

    We follow the method given by Achlioptas 2001 which guarantees that
    pairwise distances will be preserved within some epsilon, and is more
    efficient than projections involving sampling from Gaussians.
    """
    R = np.random.choice([-1, 0, 0, 0, 0, 1], (A.shape[1], k))
    return np.dot(A, R * np.sqrt(3))


@jit
def logsumexp(y):
    """Computes the log of the sum of exponentials of y in a numerically stable
    way. Useful for computing sums in log space.
    """
    ymax = y.max()
    return ymax + np.log((np.exp(y - ymax)).sum())


def sample_categorical(counts):
    """Samples from a categorical distribution parameterized by unnormalized
    counts. The index of the sampled category is returned.
    """
    sample = np.random.uniform(0, sum(counts))
    for key, count in enumerate(counts):
        if sample < count:
            return key
        sample -= count
    raise ValueError(counts)


class memoize(object):
    """Decorator for memoizing a function."""

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if args not in self.cache:
            self.cache[args] = self.func(*args)
        return self.cache[args]


def pickle_cache(pickle_path):
    """Decorating for caching a parameterless function to disk via pickle"""
    def _decorator(data_func):
        @functools.wraps(data_func)
        def _wrapper():
            if os.path.exists(pickle_path):
                return pickle.load(open(pickle_path, 'rb'))
            data = data_func()
            pickle.dump(data, open(pickle_path, 'wb'))
            return data
        return _wrapper
    return _decorator

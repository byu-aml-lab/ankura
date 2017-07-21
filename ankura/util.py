"""A collection of utility functions used throughout Ankura"""

import numpy


def random_projection(A, k):
    """Randomly projects the points of A into k-dimensions, following the
    method given by Achlioptas 2001.
    """
    R = numpy.random.choice([-1, 0, 0, 0, 0, 1], (A.shape[1], k))
    return numpy.dot(A, R * numpy.sqrt(3))


def logsumexp(y):
    """Computes the log of the sum of exponentials of y in a numerically stable
    way. Useful for computing sums in log space.
    """
    ymax = y.max()
    return ymax + numpy.log((numpy.exp(y - ymax)).sum())


def sample_categorical(counts):
    """Samples from a categorical distribution parameterized by unnormalized
    counts. The index of the sampled category is returned.
    """
    sample = numpy.random.uniform(0, sum(counts))
    for key, count in enumerate(counts):
        if sample < count:
            return key
        sample -= count
    raise ValueError(counts)


class memoize(object): # pylint: disable=invalid-name
    """Decorator for memoizing a function"""

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if args not in self.cache:
            self.cache[args] = self.func(*args)
        return self.cache[args]

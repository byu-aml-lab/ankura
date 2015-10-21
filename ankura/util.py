"""A collection of utility functions which may be useful with ankura"""

import collections
import os
import pickle

def pickle_cache(pickle_path):
    """Decorator to cache a parameterless function call to disk"""
    def _cache(data_func):
        def _load_data():
            if os.path.exists(pickle_path):
                return pickle.load(open(pickle_path))
            else:
                data = data_func()
                pickle.dump(data, open(pickle_path, 'w'))
                return data
        return _load_data
    return _cache


class memoize(object): # pylint: disable=invalid-name
    """Decorator to memoize a function"""

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if args not in self.cache:
            self.cache[args] = self.func(*args)
        return self.cache[args]


def _iscontainer(data):
    return (isinstance(data, collections.Iterable) and
            not isinstance(data, basestring))


def tuplize(data, conversion=None):
    """Converts containers into tuples, with optional data conversion"""
    if _iscontainer(data):
        return tuple(tuplize(subdata, conversion) for subdata in data)
    elif conversion:
        return conversion(data)
    else:
        return data

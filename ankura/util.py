"""A collection of utility functions which may be useful with ankura"""

import collections
import os
import pickle
import random
import tempfile


def pickle_cache(pickle_path):
    """Decorator to cache a parameterless function call to disk"""
    def _cache(data_func):
        def _load_data():
            if os.path.exists(pickle_path):
                return pickle.load(open(pickle_path, 'rb'))
            else:
                data = data_func()
                pickle.dump(data, open(pickle_path, 'wb'))
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


def named_pickle_cache(pickle_path_format):
    """Decorator to cache a single parameter function call result to disk"""
    def _cache(data_func):
        def _load_data(param):
            pickle_path = pickle_path_format.format(param)
            if os.path.exists(pickle_path):
                return pickle.load(open(pickle_path, 'rb'))
            else:
                data = data_func(param)
                ensure_dir(os.path.dirname(pickle_path))
                pickle.dump(data, open(pickle_path, 'wb'))
                return data
        return _load_data
    return _cache


def _iscontainer(data):
    return isinstance(data, collections.Iterable) and not isinstance(data, str)


def tuplize(data, conversion=None):
    """Converts containers into tuples, with optional data conversion"""
    if _iscontainer(data):
        return tuple(tuplize(subdata, conversion) for subdata in data)
    elif conversion:
        return conversion(data)
    else:
        return data


def sample_categorical(counts):
    """Sample a categorical distribution parameterized by unnormalized counts"""
    sample = random.uniform(0, sum(counts))
    for key, count in enumerate(counts):
        if sample < count:
            return key
        sample -= count

    raise ValueError(counts)


def ensure_dir(dirname):
    """Creates the given directory if it does not already exist"""
    try:
        os.makedirs(dirname)
    except FileExistsError:
        pass


def open_unique(prefix='', dirname=os.path.curdir):
    """Opens a uniquely named file

    A prefix can optionally be given to the newly created file. By default, the
    file will be created in the current directory, but this can be overriden by
    specifying a dirname. If the specified directory does not exist, it will be
    created.
    """
    ensure_dir(dirname)
    return tempfile.NamedTemporaryFile(mode='w',
                                       delete=False,
                                       prefix=prefix,
                                       dir=dirname)

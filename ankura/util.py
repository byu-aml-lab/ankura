"""Utility functions used throughout ankura"""

import functools
import pickle
import os


def pickle_cache(pickle_path):
    """Decorator to cache a parameterless function call to disk"""
    def _cache(data_func):
        @functools.wraps(data_func)
        def _load_data():
            if os.path.exists(pickle_path):
                return pickle.load(open(pickle_path, 'rb'))
            else:
                data = data_func()
                pickle.dump(data, open(pickle_path, 'wb'))
                return data
        return _load_data
    return _cache


def memoize(func):
    """Decorator to memoize a function"""
    cache = {}
    @functools.wraps(func)
    def _memoized(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return _memoized

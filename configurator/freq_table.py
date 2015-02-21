"""Frequency table.
"""

from functools import reduce
from operator import mul

import numpy as np
import pylru


class FrequencyTable(object):
    """Multivariate frequency table.

    Compute multivariate frequencies and conditional probabilities
    from a two-dimensional numpy array. Each column is expected to
    represent a discrete variable and each row a multivariate
    observation.

    Arguments:
        sample: A two-dimensional numpy array.
        domains: A list with one entry for each variable containing an
            enumerable with all the possible values of the variable.
        cache_size: Size of the LRU cache for the frequencies.

    All the arguments are available as instance attributes.
    """

    def __init__(self, domains, sample, cache_size=0):
        self.domains = domains
        self.sample = sample
        self.cache_size = cache_size
        if self.cache_size > 0:
            self._cache = pylru.lrucache(self.cache_size)

    def count_freq(self, x):
        """Count the occurences of the variable assignment in x.

        Arguments:
            x: A dictionary mapping variable indices to their values.

        Returns:
            The number of occurences of the values in x.
        """
        if self.sample is None:
            return 0
        # Return from the cache, if available.
        if self.cache_size > 0:
            cache_key = hash(frozenset(x.items()))
            if cache_key in self._cache:
                return self._cache[cache_key]
        # Compute if not available or cache is disabled.
        cumul_filter = np.ones(self.sample.shape[0], dtype=np.bool)
        for var_index, var_value in x.items():
            var_filter = self.sample[:, var_index] == var_value
            cumul_filter = np.logical_and(cumul_filter, var_filter)
        freq = cumul_filter.sum()
        # Store in the cache (if enabled) and return.
        if self.cache_size > 0:
            self._cache[cache_key] = freq
        return freq

    def cond_prob(self, x, y, add_one_smoothing=True):
        """Conditional probability distribution.

        Arguments:
            x: A dictionary mapping variable indices to their values.
            y: A dictionary mapping variable indices to their values.
            add_one_smoothing: Use add-one (Laplace) smoothing.

        Returns:
            The conditional probability of x given y.
        """
        if self.sample is None and not add_one_smoothing:
            raise ZeroDivisionError
        z = dict(x.items() | y.items())
        num = self.count_freq(z)
        den = self.count_freq(y)
        if add_one_smoothing:
            num += 1
            den += reduce(mul, (len(self.domains[i]) for i in x.keys()))
        prob = num / den
        return prob

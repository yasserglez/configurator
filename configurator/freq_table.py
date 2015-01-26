"""Frequency Table"""

from functools import reduce
from operator import mul

import numpy as np
import pylru


class FrequencyTable(object):
    """Multivariate frequency table.

    Compute multivariate frequencies and conditional probabilities
    from a 2-dimensional numpy array. Each column is expected to
    represent a categorical variable and each row a multivariate
    observation.

    Attributes:
        var_sample: A 2-dimensional numpy array.
        var_values: A list with one entry for each variable,
            containing an enumerable with all the possible values of
            the variable.
    """

    def __init__(self, var_sample, var_values, cache_size=0):
        """Initialize a new instance.

        Arguments:
            var_sample: A 2-dimensional numpy array.
            var_values: A list with one entry for each variable,
                containing an enumerable with all the possible values
                of the variable.
            cache_size: Size of the LRU cache for the frequencies.
                It is disabled by default, i.e. set to zero.
        """
        self.var_sample = var_sample
        self.var_values = var_values
        self._cache_size = cache_size
        if self._cache_size > 0:
            self._cache = pylru.lrucache(self._cache_size)

    def count_freq(self, x):
        """Count the occurences of the sample of the variables in x.

        Arguments:
            x: A dict mapping variable indices to their values.

        Returns:
            The number of occurences of the values in x.
        """
        # Return from the cache, if available.
        if self._cache_size > 0:
            cache_key = hash(frozenset(x.items()))
            if cache_key in self._cache:
                return self._cache[cache_key]
        # Compute if not available or cache is disabled.
        cumul_filter = np.ones(self.var_sample.shape[0], dtype=np.bool)
        for var_index, var_value in x.items():
            var_filter = self.var_sample[:, var_index] == var_value
            cumul_filter = np.logical_and(cumul_filter, var_filter)
        freq = cumul_filter.sum()
        # Store in the cache (if enabled) and return.
        if self._cache_size > 0:
            self._cache[cache_key] = freq
        return freq

    def cond_prob(self, x, y, add_one_smoothing=True):
        """Conditional probability distribution.

        Arguments:
            x: A dict mapping variable indices to their values.
            y: A dict mapping variable indices to their values.
            add_one_smoothing: Use add-one (Laplace) smoothing
                (default: True).

        Returns:
            The conditional probability of x given y.
        """
        z = dict(x.items() | y.items())
        num = self.count_freq(z)
        den = self.count_freq(y)
        if add_one_smoothing:
            num += 1
            x_card = [len(self.var_values[i]) for i in x.keys()]
            den += reduce(mul, x_card)
        prob = num / den
        return prob

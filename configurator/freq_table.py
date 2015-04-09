#
#  Copyright 2015 Yasser Gonzalez Fernandez
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#

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
        var_domains: A list with one entry for each variable containing
            a sequence with all the possible values of the variable.
        cache_size: Size of the LRU cache for the frequencies.

    All the arguments are available as instance attributes.
    """

    def __init__(self, var_domains, sample, cache_size=0):
        self.var_domains = var_domains
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
            if not cumul_filter.any():
                break  # all zero, no need to continue
        freq = cumul_filter.sum()
        # Store in the cache (if enabled) and return.
        if self.cache_size > 0:
            self._cache[cache_key] = freq
        return freq

    def cond_prob(self, x, y):
        """Conditional probability distribution.

        Arguments:
            x: A dictionary mapping variable indices to their values.
            y: A dictionary mapping variable indices to their values.

        Returns:
            The conditional probability of x given y.
        """
        if self.sample is None:
            # Uniform probability distribution.
            num = 1
            den = reduce(mul, (len(self.var_domains[i]) for i in x.keys()))
        else:
            z = dict(x.items() | y.items())
            num = self.count_freq(z)
            den = self.count_freq(y)
        prob = num / den
        return prob

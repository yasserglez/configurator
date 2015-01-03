"""
Frequency Table and Conditional Probabilities
"""

import numpy as np
import pylru


class FrequencyTable(object):
    """Compute frequency counts and conditional probabilities.

    Compute multivariate frequency counts and conditional
    probabilities from a 2-dimensional numpy array. Each column is
    expected to represent a categorical variable and each row a
    multi-variate observation.

    Attributes:
        data: A 2-dimensional numpy array.
    """

    def __init__(self, data, cache_size=0):
        """Initialize a new instance.

        Arguments:
            data: A 2-dimensional numpy array.
            cache_size: Set the size of a LRU cache for the frequency
                counts. It is disabled by default, i.e. set to zero.
        """
        self.data = data
        self._cache_size = cache_size
        if self._cache_size > 0:
            self._cache = pylru.lrucache(self._cache_size)

    def count_freq(self, x):
        """Count the occurences of the sample of the variables in x.

        Arguments:
            x: A dictionary mapping variable indexes to their values.

        Returns:
            The number of occurences of the values in x.
        """
        # Return from the cache, if available.
        if self._cache_size > 0:
            cache_key = hash(frozenset(x.items()))
            if cache_key in self._cache:
                return self._cache[cache_key]
        # Compute if not available or cache is disabled.
        cumul_filter = np.ones(self.data.shape[0], dtype=np.bool)
        for var_index, var_value in x.items():
            var_filter = self.data[:, var_index] == var_value
            cumul_filter = np.logical_and(cumul_filter, var_filter)
        count_freq = self.data[cumul_filter].shape[0]
        # Store in the cache (if enabled) and return.
        if self._cache_size > 0:
            self._cache[cache_key] = count_freq
        return count_freq

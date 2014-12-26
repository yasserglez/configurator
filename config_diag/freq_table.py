"""
Frequency Table and Conditional Probabilities
"""

import numpy as np


class FrequencyTable:
    """Compute frequency counts and conditional probabilities.

    A wrapper for a 2-dimensional numpy array with methods to compute
    multivariate frequency counts and conditional probabilities. The
    columns are the variables and each row is an observation.

    Attributes:
        data: A 2-dimensional numpy array.
    """
    
    def __init__(self, data):
        """Initialize a new instance.

        Arguments:
            data: A 2-dimensional numpy array.
        """
        self.data = data

    def freq(self, x):
        """Count the occurences of the sample of the variables in x.

        Arguments:
            x: A dictionary mapping column indexes to their values.

        Returns:
            The number of occurences of the values in x.
        """
        global_filter = np.ones(self.data.shape[0], dtype=np.bool)
        for col_index, col_value in x.items():
            col_filter = self.data[:, col_index] == col_value
            global_filter = np.logical_and(global_filter, col_filter)
        freq = self.data[global_filter].shape[0]
        return freq

    def joint_prob(self, x):
        """Joint probability distribution of x.

        Arguments:
            x: A dictionary mapping column indexes to their values.

        Returns:
            Probability value in [0,1].
        """
        prob = self.freq(x) / self.data.shape[0]
        return prob

    def cond_prob(self, x, y):
        """Conditional probability distribution of x given y.

        Arguments:
            x: A dictionary mapping column indexes to their values.
            y: A dictionary mapping column indexes to their values.
        
        Returns:
            Conditional probability value in [0,1].
        """
        z = dict(x.items() | y.items())
        prob = self.joint_prob(z) / self.joint_prob(y)
        return prob

"""
Frequency Table and Conditional Probabilities
"""

import numpy as np


class FrequencyTable(object):
    """Compute frequency counts and conditional probabilities.

    A wrapper for a pandas DataFrame that provides methods to compute
    multivariate frequency counts and conditional probabilities.

    Attributes:
        df: A pandas Dataframe.
    """
    
    def __init__(self, df):
        """Initialize a new instance.

        Arguments:
            df: A pandas DataFrame.
        """
        self.df = df

    def freq(self, x):
        """Count the occurence of the sample of the variables in x.

        Arguments:
            x: A dictionary mapping some variable names to their values.

        Returns:
            The number of occurences in the DataFrame of the values in x.
        """
        global_filter = np.ones(len(self.df.index), dtype=np.bool)
        for col_name, col_value in x.items():
            col_filter = self.df[col_name] == col_value
            global_filter = np.logical_and(global_filter, col_filter)
        return len(self.df[global_filter].index)

    def joint_prob(self, x):
        """Joint probability distribution of x.

        Arguments:
            x: A dictionary mapping some variable names to their values.

        Returns:
            Probability value in [0,1].
        """
        prob = self.freq(x) / len(self.df.index)
        return prob

    def cond_prob(self, x, y):
        """Conditional probability distribution of x given y.

        Arguments:
            x: A dictionary mapping some variable names to their values.
            y: A dictionary mapping some variable names to their values.
        
        Returns:
            Conditional probability value in [0,1].
        """
        z = dict(x.items() | y.items())
        prob = self.joint_prob(z) / self.joint_prob(y)
        return prob

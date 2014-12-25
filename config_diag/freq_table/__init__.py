# -*- coding: utf-8 -*-

"""
Frequency Table and Conditional Probabilities
"""


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

    def joint_prob(self, x):
        """Joint probability distribution of x.

        Arguments:
            x: A dictionary mapping some variable names to their values.

        Returns:
            Probability value in [0,1].
        """

    def cond_prob(self, x, y):
        """Conditional probability distribution of x given y.

        Arguments:
            x: A dictionary mapping some variable names to their values.
            y: A dictionary mapping some variable names to their values.
        
        Returns:
            Conditional probability value in [0,1].
        """

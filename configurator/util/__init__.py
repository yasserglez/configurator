#
#  Copyright 2015 Yasser Gonzalez Fernandez
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#

"""Utility functions.
"""

import itertools

from .c2o import load_C2O
from .sxfm import load_SXFM

__all__ = ["load_C2O", "load_SXFM"]


def iter_config_states(var_domains, exclude_terminals=False):
    """Iterate through all configuration states.

    Arguments:
         var_domains: A list with one entry for each variable containing
            a sequence with all the possible values of the variable.
        exclude_terminals: Exclude states where all the variables are known.
    """
    extended_var_values = [[None] + var_values for var_values in var_domains]
    for state_values in itertools.product(*extended_var_values):
        state = {var_index: var_value
                 for var_index, var_value in enumerate(state_values)
                 if var_value is not None}
        if len(state) != len(var_domains) or not exclude_terminals:
            yield state


def get_var_domains(sample):
    """Get the possible values of the variables from a sample.

    Arguments:
        sample: A two-dimensional numpy array containing a sample of
            the configuration variables.

    Returns:
        A list with one entry for each variable containing a
        sequence with all the possible values of the variable.
    """
    # sorted() called to return a stable order (it could be unstable
    # because of Python's hash randomization).
    var_domains = [list(sorted(set(sample[:, i])))
                   for i in range(sample.shape[1])]
    return var_domains

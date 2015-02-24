"""Utility functions.
"""

import itertools


__all__ = []


def iter_config_states(domain, exclude_terminals=False):
    """Iterate through all configuration states.

    Arguments:
         domain: A list with one entry for each variable containing
            a sequence with all the possible values of the variable.
        exclude_terminals: Exclude states where all the variables are known.
    """
    extended_var_values = [[None] + var_values for var_values in domain]
    for state_values in itertools.product(*extended_var_values):
        state = {var_index: var_value
                 for var_index, var_value in enumerate(state_values)
                 if var_value is not None}
        if len(state) != len(domain) or not exclude_terminals:
            yield state


def get_domain(sample):
    """Get the possible values of the variables from a sample.

    Arguments:
        sample: A two-dimensional numpy array containing a sample of
            the configuration variables.

    Returns:
        A list with one entry for each variable containing a
        sequence with all the possible values of the variable.
    """
    # sorted() called to return a stable order (it would be random
    # otherwise because of Python's hash randomization).
    domain = [list(sorted(set(sample[:, i])))
              for i in range(sample.shape[1])]
    return domain

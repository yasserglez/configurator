"""Configuration dialogs based on optimization.
"""

from .base import Dialog


__all__ = []


class PermutationDialog(Dialog):
    """Configuration dialog based on a permutation of the variables.

    Arguments:
        var_perm: A list containing a permutation of the variables.

    See Dialog for information about the remaining arguments,
    attributes and methods.
    """

    def __init__(self, config_values, rules, var_perm, validate=False):
        self._var_perm = var_perm
        super().__init__(config_values, rules, validate=validate)

    def _validate(self):
        if len(set(self._var_perm)) != len(self._var_perm):
            raise ValueError("Invalid var_perm value")

    def reset(self):
        super().reset()
        self._curr_var_index = 0

    def get_next_question(self):
        if not self.is_complete():
            while self._var_perm[self._curr_var_index] in self.config:
                self._curr_var_index += 1
        next_question = self._var_perm[self._curr_var_index]
        return next_question

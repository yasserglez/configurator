"""Sequence-Based Configuration Dialogs"""

from .base import Dialog


class SequenceDialog(Dialog):
    """Configuration dialog based on a permutation of the variables.

    Attributes:
        rules: A list of AssociationRule instances.
        var_seq: A list containing a permutation of the variables.

    See Dialog for other attributes.
    """

    def __init__(self, config_values, rules, var_seq, validate=False):
        """Initialize a new instance.

        Arguments:
            rules: A list of AssociationRule instances.
            var_seq: A list containing a permutation of the variables.

        See Dialog for the remaining arguments.
        """
        self.rules = rules
        self.var_seq = var_seq
        super().__init__(config_values, validate=validate)

    def _validate(self):
        if len(set(self.var_seq)) != len(self.var_seq):
            raise ValueError("Invalid var_seq value")

    def reset(self):
        """Reset the configurator to the initial state.

        See Dialog for more information.
        """
        super().reset()
        self._curr_var_index = 0

    def set_answer(self, var_index, var_value):
        """Set the value of a configuration variable.

        See Dialog for more information.
        """
        super().set_answer(var_index, var_value)
        for rule in self.rules:
            if rule.is_applicable(self.config):
                rule.apply_rule(self.config)

    def get_next_question(self):
        """Get the question that should be asked next.
        """
        if not self.is_complete():
            while self.var_seq[self._curr_var_index] in self.config:
                self._curr_var_index += 1
        next_question = self.var_seq[self._curr_var_index]
        return next_question

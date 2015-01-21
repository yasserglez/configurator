"""Sequence-Based Configuration Dialogs"""

from .base import ConfigDialog


class SequenceConfigDialog(ConfigDialog):
    """Configuration dialog based on a sequene of the variables.

    Attributes:
        rules: A list of AssociationRule instances.
        var_seq: An list containing a permutation of the variables.

    See ConfigDialog for other attributes.
    """

    def __init__(self, config_values, rules, var_seq):
        """Initialize a new instance.

        Arguments:
            rules: A list of AssociationRule instances.
            var_seq: An list containing a permutation of the variables.

        See ConfigDialog for the remaining arguments.
        """
        super().__init__(config_values)
        self.rules = rules
        self.var_seq = var_seq

    def reset(self):
        """Reset the configurator to the initial state.

        See ConfigDialog for more information.
        """
        super().reset()
        self._curr_var_index = 0

    def set_answer(self, var_index, var_value):
        """Set the value of a configuration variable.

        See ConfigDialog for more information.
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

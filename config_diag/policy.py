"""
Configuration Dialogs Based on Policies
"""

from . import ConfigDialog, ConfigDialogBuilder
from .mdp import MDP, EpisodicMDP, PolicyIteration, ValueIteration


class PolicyConfigDialog(ConfigDialog):
    """Adaptive configuration dialog based on a policy.
    """

    def __init__(self, policy):
        """Initialize a new instance.

        Arguments:
            policy: An MDP policy, i.e. a dict mapping state indexes
                to action indexes.
        """
        super().__init__()
        self._policy = policy


class MDPDialogBuilder(ConfigDialogBuilder):
    """Adpative configuration dialog builder using MDPs.
    """

    def __init__(self, config_sample=None,
                 assoc_rule_algorithm=None,
                 assoc_rule_min_support=None,
                 assoc_rule_min_confidence=None,
                 mdp_algorithm=None,
                 mdp_max_iter=None):
        """Initialize a new instance.

        Arguments:
            mdp_algorithm: Algorithm for solving the MDP. Possible
                values are: 'policy-iteration' and 'value-iteration'.
            mdp_max_iter: The maximum number of iterations of the
                algorithm used to solve the MDP.

        See ConfigDialogBuilder for the remaining arguments.
        """
        super().__init__(config_sample,
                         assoc_rule_algorithm,
                         assoc_rule_min_support,
                         assoc_rule_min_confidence)
        if mdp_algorithm == "policy-iteration":
            self._solver = PolicyIteration(max_iter=mdp_max_iter)
        elif mdp_algorithm == "value-iteration":
            self._solver = ValueIteration(max_iter=mdp_max_iter)
        else:
            raise ValueError("Invalid mdp_algorithm value")

    def build_dialog(self):
        """Construct an adaptive configuration dialog.

        Returns:
            A PolicyConfigDialog instance.
        """
        mdp = self._build_mdp()
        policy = self._solver.solve(mdp)
        dialog = PolicyConfigDialog(policy)
        return dialog

    def _build_mdp():
        pass

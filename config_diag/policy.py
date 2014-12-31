"""
Configuration Dialogs Based on Policies
"""

from . import ConfigDialog, ConfigDialogBuilder
from .mdp import ValueIteration, PolicyIteration


class PolicyConfigDialog(ConfigDialog):
    """Adaptive configuration dialog based on a policy.
    """

    def __init__(self):
        """Initialize a new instance.
        """
        super().__init__()


class MDPDialogBuilder(ConfigDialogBuilder):
    """Configuration dialog construction based on MDPs.
    """

    def __init__(self):
        """Initialize a new instance.
        """
        super().__init__()

"""
Constructing Adaptive Configuration Dialogs using Crowd Data
"""

import os
import subprocess
from collections import defaultdict
from functools import reduce

from sortedcontainers import SortedListWithKey

from .assoc_rules import AssociationRuleMiner


def get_version():
    """Get the version string.

    Returns:
        The version string.
    """
    pkg_dir = os.path.dirname(__file__)
    src_dir = os.path.abspath(os.path.join(pkg_dir, os.pardir))
    git_dir = os.path.join(src_dir, ".git")
    try:
        # Try to get the  version string dynamically from Git.
        git_args = ("git", "--work-tree", src_dir, "--git-dir", git_dir,
                    "describe", "--tags", "--dirty")
        with open(os.devnull, "w") as devnull:
            version = subprocess.check_output(git_args, stderr=devnull).strip()
    except subprocess.CalledProcessError:
        # Overwritten by a custom 'python setup.py sdist/build' command.
        version = None
    return version

__version__ = get_version()


class ConfigDialog(object):
    """Adaptive configuration dialog.

    This is the base class of all the configuration dialogs defined in
    the package (not intented to be instantiated directly). It defines
    a common interface followed by the dialogs generated using the
    different ConfigDiagBuilder subclasses.
    """

    def __init__(self):
        """Initialize a new instance.
        """
        super().__init__()


class ConfigDialogBuilder(object):
    """Adaptive configuration dialog builder.
    """

    def __init__(self, config_sample=None,
                 assoc_rule_algorithm=None,
                 assoc_rule_min_support=None,
                 assoc_rule_min_confidence=None):
        """Initialize a new instance.

        Arguments:
            config_sample: A 2-dimensional numpy array containing a
                sample of the configuration variables.
            assoc_rule_algorithm: Algorithm for mining the frequent item sets.
                Possible values are: 'apriori' and 'fp-growth'.
            assoc_rule_min_supporte: Minimum item set support in [0,1].
            assoc_rule_min_confidence: Minimum confidence of the rules [0,1].
        """
        super().__init__()
        self._config_sample = config_sample
        self._assoc_rule_algorithm = assoc_rule_algorithm
        self._assoc_rule_min_support = assoc_rule_min_support
        self._assoc_rule_min_confidence = assoc_rule_min_confidence

    def build_dialog():
        """Construct an adaptive configuration dialog.

        Returns:
            An instance of a ConfigDialog subclass.
        """
        raise NotImplementedError()

    def _mine_assoc_rules(self, min_support, min_confidence, algorithm):
        # Mine the association rules.
        miner = AssociationRuleMiner(self._config_sample)
        rules = miner.mine_assoc_rules(self._assoc_rule_min_support,
                                       self._assoc_rule_min_confidence,
                                       algorithm=self._assoc_rule_algorithm)
        # Merge rules with the same lhs. If two rules have
        # contradictory rhs, the rule with the greatest confidence
        # takes precedence.
        rule_sort_key = lambda rule: rule.confidence  # by inc. confidence
        rules_dict = defaultdict(lambda: SortedListWithKey(key=rule_sort_key))
        for rule in rules:
            lhs_key = hash(frozenset(rule.lhs.items()))
            rules_dict[lhs_key].add(rule)
        def merge_rules(r1, r2):
            r1.support = None
            r1.confidence = None
            r1.rhs.update(r2.rhs)
            return r1
        merged_rules = [reduce(merge_rules, grouped_rules)
                        for grouped_rules in rules_dict.values()]
        # Return the merged rules.
        return merged_rules

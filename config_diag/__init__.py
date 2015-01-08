"""
Constructing Adaptive Configuration Dialogs using Crowd Data
"""

import os
import logging
import subprocess
from collections import defaultdict
from functools import reduce
from operator import mul

from sortedcontainers import SortedSet, SortedListWithKey

from .freq_table import FrequencyTable
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
                 config_values=None,
                 assoc_rule_algorithm="apriori",
                 assoc_rule_min_support=None,
                 assoc_rule_min_confidence=None):
        """Initialize a new instance.

        Arguments:
            config_sample: A 2-dimensional numpy array containing a
                sample of the configuration variables.
            config_values: A list with one entry for each variable
                containing a list with all the possible values of the
                variable. If it is not given, it is automatically
                computed from the columns of config_sample.
            assoc_rule_algorithm: Algorithm for mining the frequent
                item sets. Possible values are: 'apriori' (default)
                and 'fp-growth'.
            assoc_rule_min_support: Minimum item set support in [0,1].
            assoc_rule_min_confidence: Minimum confidence in [0,1].
        """
        super().__init__()
        self._config_sample = config_sample
        self._freq_tab = FrequencyTable(self._config_sample, cache_size=0)
        if config_values is None:
            config_values = [list(SortedSet(self._config_sample[:, i]))
                             for i in range(self._config_sample.shape[1])]
        self._config_values = config_values
        self._assoc_rule_algorithm = assoc_rule_algorithm
        self._assoc_rule_min_support = assoc_rule_min_support
        self._assoc_rule_min_confidence = assoc_rule_min_confidence
        self._logger = logging.getLogger(self.__class__.__name__)

    def build_dialog():
        """Construct an adaptive configuration dialog.

        Returns:
            An instance of a ConfigDialog subclass.
        """
        raise NotImplementedError()

    def _cond_prob(self, x, y, add_one_smoothing=True):
        # Conditional probability distributionn of x given y in the
        # sample of the configuration variables. By default the
        # frequencies are computed using add-one (Laplace) smoothing.
        z = dict(x.items() | y.items())
        num = self._freq_tab.count_freq(z)
        den = self._freq_tab.count_freq(y)
        if add_one_smoothing:
            num += 1
            x_card = [len(self._config_values[i]) for i in x.keys()]
            den += reduce(mul, x_card)
        prob = num / den
        return prob

    def _mine_rules(self):
        # Mine the association rules.
        self._logger.debug("mining association rules")
        miner = AssociationRuleMiner(self._config_sample)
        rules = miner.mine_assoc_rules(self._assoc_rule_min_support,
                                       self._assoc_rule_min_confidence,
                                       algorithm=self._assoc_rule_algorithm)
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug("found %d rules", len(rules))
            supp = [rule.support for rule in rules]
            self._logger.debug("support values in [%.2f,%.2f]",
                               min(supp), max(supp))
            conf = [rule.confidence for rule in rules]
            self._logger.debug("confidence values in [%.2f,%.2f]",
                               min(conf), max(conf))
        # Merge rules with the same lhs. If two rules have
        # contradictory rhs, the rule with the greatest confidence
        # takes precedence.
        rule_sort_key = lambda rule: rule.confidence  # by inc. confidence
        rules_dict = defaultdict(lambda: SortedListWithKey(key=rule_sort_key))
        for rule in rules:
            lhs_key = hash(frozenset(rule.lhs.items()))
            rules_dict[lhs_key].add(rule)
        merged_rules = [reduce(self._merge_rules, grouped_rules)
                        for grouped_rules in rules_dict.values()]
        self._logger.debug("turned into %d rules after merging",
                           len(merged_rules))
        self._logger.debug("finished mining association rules")
        # Return the merged rules.
        return merged_rules

    def _merge_rules(self, rule1, rule2):
        # Merge two association rules with the same lhs.
        # rule1 is overwritten and returned.
        rule1.support = None
        rule1.confidence = None
        rule1.rhs.update(rule2.rhs)
        return rule1

    def _is_rule_lhs_compatible(self, rule, state):
        # Check if the rule's lhs is compatible with the given
        # configuration state. All the variables in the lhs have to
        # match the values in the state.
        return set(rule.lhs.items()) <= set(state.items())

    def _is_rule_rhs_compatible(self, rule, state):
        # Check if the rule's rhs is compatible with the given
        # configuration state. Each variable in the rhs has the same
        # value in the state or it's value is still unknown. At least
        # one variable must be unknown in order to apply the rule.
        one_unknown_var = False
        for var_index, var_value in rule.rhs.items():
            if var_index in state:
                if state[var_index] != var_value:
                    return False
            else:
                one_unknown_var = True
        return one_unknown_var

    def _is_rule_applicable(self, rule, state):
        # Check if a rule can be applied at a given configuration state.
        return (self._is_rule_lhs_compatible(rule, state) and
                self._is_rule_rhs_compatible(rule, state))

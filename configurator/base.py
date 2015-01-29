"""Base Configuration Dialogs

These classes are not intented to be instantiated directly, see
configurator.policy and configurator.sequence instead.
"""

import math
import pprint
import logging
from collections import defaultdict
from functools import reduce
from operator import mul

from sortedcontainers import SortedListWithKey

from .util import get_config_values
from .freq_table import FrequencyTable
from .assoc_rules import AssociationRuleMiner


log = logging.getLogger(__name__)


class Dialog(object):
    """Base configuration dialog.

    This is the base class of all the configuration dialogs defined in
    the package (not intented to be instantiated directly). It defines
    a common interface followed by the dialogs generated using the
    different Dialog subclasses.

    The interaction with all subclasses must be as follows. First, the
    reset method should be called to begin at a state where all the
    configuration variables are unknown. Next, a call to the method
    get_next_question will suggest a question, which can be posed to
    the user and the answer should be given as feedback to the dialog
    using the set_answer method. It is possible to ignore the
    suggestion given by the dialog and answer the questions in any
    order. In this case, simply call set_answer and future calls to
    get_next_question will act accordingly.

    The config attribute can be used at any time to retrieve the
    configuration values collected so far. The is_complete method can
    be used to check whether all the variables has been set.

    Attributes:
        config_values: A list with one entry for each variable,
            containing an enumerable with all the possible values of
            the variable.
        config: The current configuration state, i.e. a dict mapping
            variable indices to their values.
    """

    def __init__(self, config_values, rules, validate=False):
        """Initialize a new instance.

        Arguments:
            config_values: A list with one entry for each variable,
                containing an enumerable with all the possible values
                of the variable.
            rules: A list of AssociationRule instances.
            validate: Indicates whether the dialog initialization
                should be validated or not (default: False).
                A ValueError exception will be raised if any error
                is found.
        """
        super().__init__()
        self.config_values = config_values
        self._rules = rules
        self.reset()
        if validate:
            self._validate()

    def _validate(self):
        pass

    def reset(self):
        """Reset the configurator to the initial state.

        In the initial configuration state the value of all the
        variables is unknown. This method must be called before making
        any call to get_next_question or set_answer methods.
        """
        self.config = {}

    def get_next_question(self):
        """Get the question that should be asked next.

        Returns the question that should be asked next to the user,
        according to this dialog. Each question is identified by the
        index of the corresponding variable.

        Returns:
            An integer, the variable index.
        """
        raise NotImplementedError

    def set_answer(self, var_index, var_value):
        """Set the value of a configuration variable.

        It wil be usually called with a variable index returned right
        before by get_next_question and the answer that the user gave
        to the question.

        Arguments:
            var_index: An integer, the variable index.
            var_value: The value of the variable. It must be one of
                the possible values of the variable in the
                config_values instance attribute.
        """
        if var_index in self.config:
            raise ValueError("Variable {0} is already set".format(var_index))
        else:
            self.config[var_index] = var_value
            for rule in self._rules:
                if rule.is_applicable(self.config):
                    rule.apply_rule(self.config)

    def is_complete(self):
        """Check if the configuration is complete.

        Returns:
            True if the values of all the variables has been set,
            False otherwise.
        """
        return len(self.config) == len(self.config_values)


class DialogBuilder(object):
    """Base configuration dialog builder.
    """

    def __init__(self, config_sample=None,
                 config_values=None,
                 validate=False,
                 assoc_rule_algorithm="apriori",
                 assoc_rule_min_support=None,
                 assoc_rule_min_confidence=None):
        """Initialize a new instance.

        Arguments:
            config_sample: A 2-dimensional numpy array containing a
                sample of the configuration variables.
            config_values: A list with one entry for each variable,
                containing an enumerable with all the possible values
                of the variable. If it is not given, it is computed
                from config_sample.
            validate: Whether or not to run some (generally costly)
                checks on the generated model and the resulting Dialog
                instance (default: False). Mostly for debugging purposes.
            assoc_rule_algorithm: Algorithm for mining the frequent
                item sets. Possible values are: 'apriori' (default)
                and 'fp-growth'.
            assoc_rule_min_support: Minimum item set support in [0,1].
            assoc_rule_min_confidence: Minimum confidence in [0,1].
        """
        super().__init__()
        self._config_sample = config_sample
        if config_values is None:
            config_values = get_config_values(config_sample)
        self._config_values = config_values
        self._freq_tab = FrequencyTable(self._config_sample,
                                        self._config_values,
                                        cache_size=1000)
        config_card = reduce(mul, map(len, self._config_values))
        log.debug("%d possible configurations of %d variables (%d binary)",
                  config_card, len(self._config_values),
                  math.ceil(math.log2(config_card)))
        log.debug("the configuration sample has %d observations",
                  self._config_sample.shape[0])
        self._validate = validate
        self._assoc_rule_algorithm = assoc_rule_algorithm
        self._assoc_rule_min_support = assoc_rule_min_support
        self._assoc_rule_min_confidence = assoc_rule_min_confidence

    def build_dialog(self):
        """Construct a configuration dialog.

        Returns:
            An instance of a Dialog subclass.
        """
        raise NotImplementedError

    def _mine_rules(self):
        # Mine the association rules.
        log.debug("mining association rules")
        miner = AssociationRuleMiner(self._config_sample)
        rules = miner.mine_assoc_rules(self._assoc_rule_min_support,
                                       self._assoc_rule_min_confidence,
                                       algorithm=self._assoc_rule_algorithm)
        if log.isEnabledFor(logging.DEBUG):
            if rules:
                log.debug("found %d rules", len(rules))
                supp = [rule.support for rule in rules]
                log.debug("support values in [%.2f,%.2f]",
                          min(supp), max(supp))
                conf = [rule.confidence for rule in rules]
                log.debug("confidence values in [%.2f,%.2f]",
                          min(conf), max(conf))
            else:
                log.debug("no rules were found")
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
        if merged_rules:
            log.debug("turned into %d rules after merging:\n%s",
                      len(merged_rules), pprint.pformat(merged_rules))
        log.debug("finished mining association rules")
        # Return the merged rules.
        return merged_rules

    def _merge_rules(self, rule1, rule2):
        # Merge two association rules with the same lhs.
        # rule1 is overwritten and returned.
        rule1.support = None
        rule1.confidence = None
        rule1.rhs.update(rule2.rhs)
        return rule1

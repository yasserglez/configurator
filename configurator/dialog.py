"""Base configuration dialogs.
"""

import math
import pprint
import logging
from collections import defaultdict
from functools import reduce
from operator import mul

import numpy as np
from sortedcontainers import SortedListWithKey

from .rules import RuleMiner
from .csp import CSP

from .util import get_config_values
from .freq_table import FrequencyTable


__all__ = ["DialogBuilder", "Dialog"]


log = logging.getLogger(__name__)


class DialogBuilder(object):
    """Base configuration dialog builder.

    This is the base class of all the configuration dialog builders
    defined in the package (it is not intented to be instantiated
    directly). See the subclasses defined in :mod:`configurator.dp`
    and :mod:`configurator.rl` for concrete dialog builders.

    The dialog builders support representing the configuration problem
    using a rule-based or a constraint-based specification (although
    all builders may not support both specifications). In both cases,
    :meth:`build_dialog` can be used to construct a configuration
    dialog that guides the users throught the configuration process
    minimizing the number of questions that must be asked in order to
    obtain a complete configuration.

    Arguments:
        domain: A list with one entry for each variable containing an
            enumerable with all the possible values of the variable.
            All the variables must be domain-consistent (i.e. there
            must exist at least one consistent configuration in which
            each value value occurs).
        rules: A list of :class:`configurator.rules.Rule` instances.
            Rules with the same left-hand-side will be combined into a
            single rule by merging their right-hand-sides (values set
            by rules found later in the list take precedence over
            values set by earlier rules).
        constraints: A list of tuples with two components each: i) a
            tuple with the indices of the variables involved in the
            constraint, and ii) a function that checks the constraint.
            The constraint functions will receive a variables tuple
            and a values tuple, both containing only the restricted
            variable indices and their values (in the same order
            provided in `constraints`). The function should return
            `True` if the values satisfy the constraint, `False`
            otherwise. The constraints must be normalized (i.e. two
            different constraints shouldn't involve the same set of
            variables).
        sample: A two-dimensional numpy array containing a sample of
            the configuration variables. Each column is expected to
            represent a discrete variable and each row a multivariate
            observation. The order of the columns must match the order
            of the variables in `domain`.
        validate: Whether or not to run some (generally costly) checks
            on the generated model and the resulting :class:`Dialog`
            instance (default: `False`). Mostly intended for testing
            purposes.

    The `domain` argument must always be given, as it defines the
    domain of the variables. The `rules` argument is used with the
    rule-based specification and the `constraints` argument with the
    constraint-based specification. In both cases, it assumed that
    there are no contradictions in the configuration problem. The
    satisfiability of a constraint-based specification can be verified
    using the :meth:`~configurator.csp.CSP.solve` method of the
    :class:`configurator.csp.CSP` class. Note that both `rules` and
    `constraints` cannot be given at the same time, but it is possible
    to express the rules as constraints if needed. If the `sample`
    argument is given, it will be considered how likely is that the
    user will select each configuration when using the dialog,
    otherwise it will be assumed that all configurations occur with
    the same probability.
    """

    def __init__(self, domain, rules=None, constraints=None, validate=False):
        super().__init__()
        self._config_sample = config_sample
        if config_values is None:
            config_values = get_config_values(config_sample)
        self._config_values = config_values
        self._freq_table = FrequencyTable(self._config_values,
                                          self._config_sample,
                                          cache_size=1000)
        config_card = reduce(mul, map(len, self._config_values))
        log.info("there are %d possible configurations of %d variables",
                 config_card, len(self._config_values))
        log.info("it is equivalent to %d binary variables",
                 math.ceil(math.log2(config_card)))
        log.info("the configuration sample has %d observations",
                 self._config_sample.shape[0])
        self._validate = validate
        self._rule_min_support = rule_min_support
        self._rule_min_confidence = rule_min_confidence

    def build_dialog(self):
        """Construct a configuration dialog.

        Returns:
            An instance of a :class:`Dialog` subclass.
        """
        raise NotImplementedError

    def _mine_rules(self):
        # Mine the association rules.
        log.info("mining association rules")
        miner = RuleMiner(self._config_sample)
        rules = miner.mine_rules(self._rule_min_support,
                                 self._rule_min_confidence,
                                 min_len=2, max_len=2)
        if rules:
            log.info("found %d rules", len(rules))
            supp = [rule.support for rule in rules]
            log.info("support five-number summary:\n%s",
                     np.array_str(np.percentile(supp, [0, 25, 50, 75, 100])))
            conf = [rule.confidence for rule in rules]
            log.info("confidence five-number summary:\n%s",
                     np.array_str(np.percentile(conf, [0, 25, 50, 75, 100])))
        else:
            log.info("no rules were found")
        # Merge rules with the same lhs. If two rules have
        # contradictory rhs, the rule with the greatest support (i.e.
        # the most popular) takes precedence.
        rule_sort_key = lambda rule: rule.support  # ascending by support
        rules_dict = defaultdict(lambda: SortedListWithKey(key=rule_sort_key))
        for rule in rules:
            lhs_key = hash(frozenset(rule.lhs.items()))
            rules_dict[lhs_key].add(rule)
        merged_rules = [reduce(self._merge_rules, grouped_rules)
                        for grouped_rules in rules_dict.values()]
        if merged_rules:
            log.info("turned into %d rules after merging", len(merged_rules))
            log.debug("merged rules:\n%s", pprint.pformat(merged_rules))
        log.info("finished mining association rules")
        # Return the merged rules.
        return merged_rules

    def _merge_rules(self, rule1, rule2):
        # Merge two association rules with the same lhs.
        # rule1 is overwritten and returned.
        rule1.support = None
        rule1.confidence = None
        rule1.rhs.update(rule2.rhs)
        return rule1


class Dialog(object):
    """Base configuration dialog.

    This is the base class of all the configuration dialogs defined in
    the package (not intented to be instantiated directly). It defines
    a common interface shared by the dialogs constructed using the
    different :class:`DialogBuilder` subclasses.

    Arguments:
        domain: A list with one entry for each variable containing an
            enumerable with all the possible values of the variable.
        rules: A list of :class:`configurator.rules.Rule` instances.
        constraints: A list of tuples with two components each: i) a
            tuple with the indices of the variables involved in the
            constraint, and ii) a function that checks the constraint.
        validate: Indicates whether the dialog initialization should
            be validated or not (default: `False`).

    All the arguments are available as instance attributes.

    The interaction with all subclasses must be as follows. First,
    :meth:`reset` should be called to begin at a state where all the
    configuration variables are unknown. Next, a call to
    :meth:`get_next_question` will suggest a question, which can be
    asked to the user and the answer should be given as feedback to
    the dialog using :meth:`set_answer`. It is possible to ignore the
    suggestion given by the dialog and answer the questions in any
    order. In this case, simply call :meth:`set_answer` and future
    calls to :meth:`get_next_question` will act accordingly.

    The :attr:`config` attribute can be used at any time to retrieve
    the configuration values collected so far. Also,
    :meth:`get_possible_answers` can be used to obtain the possible
    answers to a question, given the current partial configuration
    (answers given back to the dialog with :meth:`set_answer` must be
    restricted to one of these values). :meth:`is_complete` can be
    used to check whether all the variables have been set.
    """

    def __init__(self, config_values, rules, validate=False):
        super().__init__()
        self.config_values = config_values
        self.rules = rules
        self.reset()
        if validate:
            self._validate()

    def _validate(self):
        pass

    def reset(self):
        """Reset the dialog to the initial state.

        In the initial configuration state the value of all the
        variables is unknown. This method must be called before making
        any call to the other methds.
        """
        self.config = {}

    def get_next_question(self):
        """Get the question that should be asked next.

        Returns the question that should be asked next to the user
        according to the dialog. Each question is identified by the
        index of the corresponding variable.

        Returns:
            An integer, the variable index.
        """
        raise NotImplementedError

    def get_possible_answers(self, var_index):
        """Get the possible answers to a question.

        Arguments:
            var_index: An integer, the variable index.

        Returns:
            A list with the possible answers.
        """

    def set_answer(self, var_index, var_value):
        """Set the value of a configuration variable.

        This method wil be usually called with a variable index
        returned right before by :meth:`get_next_question` and the
        answer that the user gave to the question.

        Arguments:
            var_index: An integer, the variable index.
            var_value: The value of the variable. It must be one of
                the possible values of the variable in the
                :attr:`config_values` attribute.
        """
        assert var_index not in self.config, \
            "Variable {0} is already set".format(var_index)
        self.config[var_index] = var_value
        for rule in self.rules:
            if rule.is_applicable(self.config):
                rule.apply_rule(self.config)

    def is_complete(self):
        """Check if the configuration is complete.

        Returns:
            `True` if the values of all the variables has been set,
            `False` otherwise.
        """
        return len(self.config) == len(self.config_values)

"""Base configuration dialogs.
"""

import pprint
import logging
from collections import defaultdict
from functools import reduce
from operator import mul

from .rules import Rule
from .csp import CSP
from .freq_table import FrequencyTable


__all__ = ["DialogBuilder", "Dialog"]


log = logging.getLogger(__name__)


class DialogBuilder(object):
    """Base configuration dialog builder.

    This is the base class of all the configuration dialog builders
    defined in the package (it is not intented to be instantiated
    directly). See the subclasses defined in :mod:`configurator.policy`
    and :mod:`configurator.sequence` for concrete dialog builders.

    The dialog builders support representing the configuration problem
    using a rule-based or a constraint-based specification (although
    all builders may not support both specifications). In both cases,
    :meth:`build_dialog` can be used to construct a configuration
    dialog that guides the users throught the configuration process
    minimizing the number of questions that must be asked in order to
    obtain a complete configuration.

    Arguments:
        var_domains: A list with one entry for each variable
            containing a sequence with all the possible values of the
            variable. All the variables must be domain-consistent.
        sample: A two-dimensional numpy array containing a sample of
            the configuration variables. Each column is expected to
            represent a discrete variable and each row a multivariate
            observation. The order of the columns must match the order
            of the variables in `var_domains`.
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
            `True` if the values satisfy the constraint. The
            constraints must be normalized (i.e. two different
            constraints shouldn't involve the same set of variables).
        validate: Whether or not to run some (generally costly) checks
            on the generated model and the resulting :class:`Dialog`
            instance. Mostly intended for testing purposes.

    The `var_domains` argument must always be given, as it defines the
    domain of the variables. The `rules` argument is used with the
    rule-based specification and `constraints` with the
    constraint-based specification. In both cases, it assumed that
    there are no contradictions in the configuration problem. The
    satisfiability of a constraint-based specification can be verified
    using the :meth:`~configurator.csp.CSP.is_consistent` method of
    the :class:`configurator.csp.CSP` class. Note that both `rules`
    and `constraints` cannot be given at the same time, but it is
    possible to express the rules as constraints. If `sample` is
    given, it defines the probabilities of the user's responses,
    otherwise it will be assumed that all valid configurations occur
    with the same probability.

    All the arguments are available as instance attributes.
    """

    def __init__(self, var_domains, sample=None, rules=None,
                 constraints=None, validate=False):
        super().__init__()
        self.var_domains = [list(var_domain) for var_domain in var_domains]
        log.info("there are %g possible configurations of %d variables",
                 reduce(mul, map(len, self.var_domains)),
                 len(self.var_domains))
        log.debug("variable domains:\n%s", pprint.pformat(self.var_domains))
        self.sample = sample
        if self.sample is not None:
            log.info("the configuration sample has %g observations",
                     self.sample.shape[0])
        # Build the frequency table from the configuration sample.
        self._freq_table = FrequencyTable(self.var_domains, self.sample,
                                          cache_size=1000)
        # Validate and process the rules and constraints.
        if not (rules or constraints):
            raise ValueError("One of rules or constraints must be given")
        if rules and constraints:
            raise ValueError("Both rules and constraints " +
                             "cannot be given at the same time")
        self.rules = rules if rules is not None else []
        if self.rules:
            log.info("using %d rules", len(self.rules))
            # Merge rules with the same lhs.
            rules_dict = defaultdict(list)
            for rule in self.rules:
                lhs_key = hash(frozenset(rule.lhs.items()))
                rules_dict[lhs_key].append(rule)
            self.rules = [reduce(self._merge_rules, grouped_rules)
                          for grouped_rules in rules_dict.values()]
            log.debug("merged rules:\n%s", pprint.pformat(self.rules))
        self.constraints = constraints
        if self.constraints is not None:
            log.info("using %d constraints", len(self.constraints))
        self._validate = validate

    def build_dialog(self):
        """Construct a configuration dialog.

        Returns:
            An instance of a `Dialog` subclass.
        """
        raise NotImplementedError

    def _merge_rules(self, rule1, rule2):
        # Merge two association rules with the same lhs.
        assert rule1.lhs == rule2.lhs
        merged_rule = Rule(rule1.lhs, rule1.rhs)
        merged_rule.rhs.update(rule2.rhs)
        return merged_rule


class Dialog(object):
    """Base configuration dialog.

    This is the base class of all the configuration dialogs defined in
    the package (not intented to be instantiated directly). It defines
    a common interface shared by the dialogs constructed using the
    different :class:`DialogBuilder` subclasses.

    Arguments:
        var_domains: A list with one entry for each variable containing
            a sequence with all the possible values of the variable.
        rules: A list of :class:`configurator.rules.Rule` instances.
        constraints: A list of tuples with two components each: i) a
            tuple with the indices of the variables involved in the
            constraint, and ii) a function that checks the constraint.
        validate: Indicates whether the dialog initialization should
            be validated or not.

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
    restricted to one of these values). :meth:`is_consistent` checks
    if the current (possibly partial) configuration is consistent
    (only relevant if using local consistency checks, because
    otherwise it will always be consistent). :meth:`is_complete` can
    be used to check whether all the variables have been set.

    All the arguments are available as instance attributes.
    """

    def __init__(self, var_domains,
                 rules=None, constraints=None,  validate=False):
        super().__init__()
        self.var_domains = var_domains
        if not (rules or constraints):
            raise ValueError("One of rules or constraints must be given")
        if rules and constraints:
            raise ValueError("Both rules and constraints " +
                             "cannot be given at the same time")
        self.rules = rules if rules is not None else []
        self.constraints = constraints
        if self.constraints is not None:
            self._csp = CSP(self.var_domains, self.constraints)
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
        if self.constraints:
            self._csp.reset()

    def get_next_question(self):
        """Get the question that should be asked next.

        Returns the question that should be asked next to the user
        according to the dialog. Each question is identified by the
        index of the corresponding variable.

        Returns:
            The variable index.
        """
        raise NotImplementedError

    def get_possible_answers(self, var_index):
        """Get the possible answers to a question.

        Arguments:
            var_index: The variable index.

        Returns:
            A list with the possible answers.
        """
        if var_index in self.config:
            raise ValueError("The question has already been answered")
        possible_answers = (self.var_domains[var_index] if self.rules else
                            self._csp.pruned_var_domains[var_index])
        return possible_answers

    def set_answer(self, var_index, var_value, consistency="global"):
        """Set the value of a configuration variable.

        This method wil be usually called with a variable index
        returned right before by :meth:`get_next_question` and the
        answer that the user gave to the question.

        Arguments:
            var_index: The variable index.
            var_value: The value of the variable. It must be one of
                the possible values of the variable returned by
                :meth:`get_possible_answers`.
             consistency: Type of consistency check used to filter the
                domain of the remaining questions. Possible values
                are: `'global'` and `'local'`. This argument is
                ignored for rule-based dialogs.
        """
        if var_index in self.config:
            raise ValueError("The question has already been answered")
        if var_value not in self.get_possible_answers(var_index):
            raise ValueError(("%r is not a valid answer for %d given " +
                              "the current partial configuration") %
                             (var_value, var_index))
        if self.rules:
            prev_config_len = len(self.config)
            self.config[var_index] = var_value
            while len(self.config) > prev_config_len:
                prev_config_len = len(self.config)
                for rule in self.rules:
                    if rule.is_applicable(self.config):
                        rule.apply_rule(self.config)
        if self.constraints:
            self._csp.assign_variable(var_index, var_value, consistency)
            self.config.update(self._csp.assignment)

    def is_consistent(self):
        """Check if the current partial configuration is consistent.

        Returns:
            `True` if it's consistent, `False` otherwise.
        """
        if self.constraints:
            for var_index in range(len(self.var_domains)):
                if var_index not in self.config:
                    possible_answers = self.get_possible_answers(var_index)
                    if not possible_answers:
                        return False
        return True

    def is_complete(self):
        """Check if the configuration is complete.

        Returns:
            `True` if all the variables has been set, `False` otherwise.
        """
        return len(self.config) == len(self.var_domains)

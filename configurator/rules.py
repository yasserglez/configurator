#
#  Copyright 2015 Yasser Gonzalez Fernandez
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#

"""Association rule mining.
"""

from fim import fpgrowth


class Rule(object):
    """Association rule.

    Arguments:
        lhs: Left-hand-side (also called antecedent or body) of the rule.
            A dictionary mapping variable indices to their values.
        rhs: Right-hand-side (also called consequent or head) of the rule.
            A dictionary mapping variable indices to their values.
        support: Rule support in [0,1] (optional).
        confidence: Rule confidence in [0,1] (optional).

    All the arguments are available as instance attributes.
    """

    def __init__(self, lhs, rhs, support=None, confidence=None):
        self.lhs = lhs
        self.rhs = rhs
        self.support = support
        self.confidence = confidence

    def __repr__(self):
        return "Rule(lhs=%r, rhs=%r)" % (self.lhs, self.rhs)

    def is_lhs_compatible(self, assignment):
        """Check left-hand-side compatibility.

        Check if the rule's left-hand-side is compatible with a
        variable assignment. All the variables in the left-hand-side
        must match the assigned values.

        Arguments:
            assignment: A dictionary mapping variable indices to
                their values.

        Returns:
            `True` if the left-hand-side is compaible, `False` otherwise.
        """
        for var_index, var_value in self.lhs.items():
            if (var_index not in assignment
                    or assignment[var_index] != var_value):
                return False
        return True

    def is_rhs_compatible(self, assignment):
        """Check right-hand-side compatibility.

        Check if the rule's right-hand-side is compatible with a
        partial assignment of the variables. Each variable in the
        right-hand-side must be unknown or have the same observed
        value. At least one variable must be unknown to avoid trivial
        applications of the rule.

        Arguments:
            assignment: A dictionary mapping variable indices to their values.

        Returns:
            `True` if the right-hand-side is compaible, `False` otherwise.
        """
        one_unknown_var = False
        for var_index, var_value in self.rhs.items():
            if var_index in assignment:
                if assignment[var_index] != var_value:
                    return False
            else:
                one_unknown_var = True
        return one_unknown_var

    def is_applicable(self, assignment):
        """Check if the rule can be applied.

        Check if both the left-hand-side and the right-hand-side of
        the rule are compatible with a partial assignment of the
        variables. See the documentation of :meth:`is_lhs_compatible`
        and :meth:`is_rhs_compatible`.

        Arguments:
            assignment: A dictionary mapping variable indices to their values.

        Returns:
            `True` if rule is applicable, `False` otherwise.
        """
        return (self.is_lhs_compatible(assignment) and
                self.is_rhs_compatible(assignment))

    def apply_rule(self, assignment):
        """Apply the rule.

        Complete the values of a partial assignment of the variables
        by setting the variables in the right-hand-side to the values
        indicated by the rule. :meth:`is_applicable` must be called
        first to ensure that no variables will be overwritten.

        Arguments:
            assignment: A dictionary mapping variable indices to their values.
                The dictionary is updated in-place.
        """
        assignment.update(self.rhs)


class RuleMiner(object):
    """Association rule mining.

    Arguments:
        sample: A two-dimensional numpy array.

    All the arguments are available as instance attributes.

    Discover association rules in a two-dimensional numpy array. Each
    column is expected to represent a discrete variable and each row a
    multivariate observation. The frequent item sets are mined using
    the FP-growth algorithm implementation provided by Christian
    Borgelt's PyFIM library available at http://www.borgelt.net/pyfim.html.
    """

    def __init__(self, sample):
        self.sample = sample
        # Transform the numpy array into PyFIM's expected input.
        self._transactions = (enumerate(self.sample[i, :])
                              for i in range(self.sample.shape[0]))

    def mine_rules(self, min_support=0.1, min_confidence=0.8,
                   min_len=2, max_len=None):
        """Discover association rules.

        Arguments:
            min_support: Minimum rule support in [0,1].
            min_confidence: Minimum rule confidence in [0,1].
            min_len: Minimum number of items per item set.
            max_len: Maximum number of items per item set (default: no limit).

        Returns:
            A list of `Rule` instances.
        """
        max_len = -1 if max_len is None else max_len
        min_support = 100 * min_support
        min_confidence = 100 * min_confidence
        result = fpgrowth(self._transactions, target="r",
                          zmin=min_len, zmax=max_len,
                          supp=min_support, conf=min_confidence,
                          report="sc", mode="o")
        rules = [Rule(dict(lhs), dict((rhs, )), report[0], report[1])
                 for rhs, lhs, report in result]
        return rules

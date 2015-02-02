"""Association Rule Mining"""

from fim import apriori, fpgrowth


class AssociationRule(object):
    """An association rule mined by AssociationRuleMiner.

    Attributes:
        lhs: Left-hand-side (also called antecedent or body) of the rule.
            A dict mapping variable indices to their values.
        rhs: Right-hand-side (also called consequent or head) of the rule.
            A dict mapping variable indices to their values.
        support: Item set support in [0,1].
        confidence: Confidence of the association rule in [0,1].
    """

    def __init__(self, lhs, rhs, support=None, confidence=None):
        """Initialize a new instance.

        Arguments:
            lhs: Left-hand-side of the rule.
            rhs: Right-hand-side of the rule.
            support: Item set support.
            confidence: Confidence of the association rule.
        """
        self.lhs = lhs
        self.rhs = rhs
        self.support = support
        self.confidence = confidence

    def __repr__(self):
        return "AssociationRule(lhs=%r, rhs=%r)" % (self.lhs, self.rhs)

    def is_lhs_compatible(self, observation):
        """Check left-hand-side compatibility.

        Check if the rules' left-hand-side is compatible with an
        observation of the variables. All the variables in the
        left-hand-side must match the observed values.

        Arguments:
            observation: A dict mapping variable indices to
                their values.

        Returns:
            True if the left-hand-side is compaible, False if not.
        """
        for var_index, var_value in self.lhs.items():
            if (var_index not in observation
                    or observation[var_index] != var_value):
                return False
        return True

    def is_rhs_compatible(self, observation):
        """Check right-hand-side compatibility.

        Check if the rules' right-hand-side is compatible with a
        partial observation of the variables. Each variable in the
        right-hand-side must be unknown or have the same observed
        value. At least one variable must be unknown to avoid trivial
        applications of the rule.

        Arguments:
            observation: A dict mapping variable indices to their values.

        Returns:
            True if the right-hand-side is compaible, False if not.
        """
        one_unknown_var = False
        for var_index, var_value in self.rhs.items():
            if var_index in observation:
                if observation[var_index] != var_value:
                    return False
            else:
                one_unknown_var = True
        return one_unknown_var

    def is_applicable(self, observation):
        """Check if the rule can be applied.

        Check if both the left-hand-side and the right-hand-side of
        the rule are compatible with a partial observation of the
        variables. See the documentation of the is_lhs_compatible and
        is_rhs_compatible methods.

        Arguments:
            observation: A dict mapping variable indices to their values.

        Returns:
            True if rule is applicable, False if not.
        """
        return (self.is_lhs_compatible(observation) and
                self.is_rhs_compatible(observation))

    def apply_rule(self, observation):
        """Apply the rule.

        Complete the values of a partial observation of the variables
        by setting the variables in the right-hand-side to the values
        indicated by the rule. is_applicable method must be called
        first to ensure that no variables will be overwritten.

        Arguments:

            observation: A dict mapping variable indices to their values.
                It is updated in-place.
        """
        observation.update(self.rhs)


class AssociationRuleMiner(object):
    """Association rule mining.

    Discover association rules in a 2-dimensional numpy array. Each
    column is expected to represent a categorical variable and each
    row a multi-variate observation. Two algorithms (Apriori and
    FP-growth) are supported for frequent item set mining, both
    implementations provided by Christian Borgelt's PyFIM library
    available at http://www.borgelt.net/pyfim.html.

    Attributes:
        data: A 2-dimensional numpy array.
    """

    def __init__(self, data):
        """Initialize a new instance.

        Arguments:
            data: A 2-dimensional numpy array.
        """
        self.data = data
        # Transform the numpy array into PyFIM's expected input.
        self._transactions = (enumerate(self.data[i, :])
                              for i in range(self.data.shape[0]))

    def mine_assoc_rules(self, min_support=0.1, min_confidence=0.8,
                         min_len=2, max_len=None, algorithm="fp-growth"):
        """
        Arguments:
            min_support: Minimum rule support in [0,1] (default: 0.1).
            min_confidence: Minimum confidence of the rules in [0,1]
                (default: 0.8).
            min_len: Minimum number of items per item set (default: 2).
            max_len: Maximum number of items per item set (default: no limit).
            algorithm: Algorithm for mining the frequent item sets.
                Possible values are: 'apriori' and 'fp-growth' (default).

        Returns:
            A list of AssociationRule instances.
        """
        if algorithm not in ("apriori", "fp-growth"):
            raise ValueError("Invalid frequent item set mining algorithm")
        max_len = -1 if max_len is None else max_len
        min_support = 100 * min_support
        min_confidence = 100 * min_confidence
        algorithm = apriori if algorithm == "apriori" else fpgrowth
        result = algorithm(self._transactions, target="r",
                           zmin=min_len, zmax=max_len,
                           supp=min_support, conf=min_confidence,
                           report="sc", mode="o")
        rules = [AssociationRule(dict(lhs), dict((rhs, )),
                                 report[0], report[1])
                 for rhs, lhs, report in result]
        return rules

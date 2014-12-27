"""
Association Rule Mining
"""

from fim import apriori, fpgrowth


class AssociationRule:
    """An association rule mined by AssociationRuleMiner.

    Attributes:
        lhs: Left-hand-side (also called antecedent or body) of the rule.
            A dictionary mapping variable indexes to their values.
        rhs: Right-hand-side (also called consequent or head) of the rule.
            A dictionary mapping variable indexes to their values.
        support: Item set support in [0,1].
        confidence: Confidence of the association rule in [0,1].
    """

    def __init__(self, lhs, rhs, support, confidence):
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


class AssociationRuleMiner:
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
                         min_len=2, max_len=None, algorithm='apriori'):
        """
        Arguments:
            min_support: Minimum item set support in [0,1] (default: 0.1).
            min_confidence: Minimum confidence of the rules in [0,1]
                (default: 0.8).
            min_len: Minimum number of items per item set (default: 2).
            max_len: Maximum number of items per item set (default: no limit).
            algorithm: Algorithm for mining the frequent item sets.
                Possible values are: 'apriori' (default) and 'fpgrowth'.

        Returns:
            A list of AssociationRule instances.
        """
        if algorithm not in ('apriori', 'fpgrowth'):
            raise ValueError('Invalid frequent item set mining algorithm')
        max_len = -1 if max_len is None else max_len
        min_support = 100 * min_support
        min_confidence = 100 * min_confidence
        algorithm = fpgrowth if algorithm == 'fpgrowth' else apriori
        result = algorithm(self._transactions, target='r', report='se',
                           zmin=min_len, zmax=max_len, supp=min_support,
                           conf=min_confidence, eval='c', thresh=0)
        rules = [AssociationRule(dict(lhs), dict((rhs, )),
                                 report[0], report[1])
                 for rhs, lhs, report in result]
        return rules

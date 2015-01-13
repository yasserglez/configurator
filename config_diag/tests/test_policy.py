import sys
import logging

from .examples import load_email_client
from ..policy import MDPDialogBuilder


logging.basicConfig(format="%(asctime)s:%(name)s:%(funcName)s:%(message)s",
                    level=logging.DEBUG)


config_sample, min_support, min_confidence, policy = load_email_client()


class TestMDPDialogBuilder(object):

    def setup(self):
        print("", file=sys.stderr)  # newline before the logging output

    def _test_builder(self, algorithm, discard_states,
                      partial_assoc_rules, collapse_terminals):
        builder = MDPDialogBuilder(
            config_sample=config_sample,
            assoc_rule_algorithm="apriori",
            assoc_rule_min_support=min_support,
            assoc_rule_min_confidence=min_confidence,
            mdp_algorithm=algorithm,
            mdp_discard_states=discard_states,
            mdp_partial_assoc_rules=partial_assoc_rules,
            mdp_collapse_terminals=collapse_terminals,
            mdp_validate=True)
        dialog = builder.build_dialog()
        initial_state = frozenset()
        assert dialog.policy[initial_state] == policy[initial_state]

    def _test_builder_without_optim(self, algorithm):
        self._test_builder(algorithm, False, False, False)

    def _test_builder_with_optim(self, algorithm):
        self._test_builder(algorithm, True, True, True)

    def test_value_iteration_without_optim(self):
        self._test_builder_without_optim("value-iteration")

    def test_policy_iteration_without_optim(self):
        self._test_builder_without_optim("policy-iteration")

    def test_value_iteration_with_optim(self):
        self._test_builder_with_optim("value-iteration")

    def test_policy_iteration_with_optim(self):
        self._test_builder_with_optim("policy-iteration")

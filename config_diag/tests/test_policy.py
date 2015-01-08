import sys
import logging

from .examples import load_titanic
from ..policy import MDPDialogBuilder


logging.basicConfig(format="%(asctime)s:%(name)s:%(funcName)s:%(message)s",
                    level=logging.DEBUG)


class TestMDPDialogBuilder(object):

    def setup(self):
        self.config_sample = load_titanic()
        print("", file=sys.stderr)  # newline before the logging output

    def _test_builder(self, algorithm, discard_states,
                      partial_assoc_rules, collapse_terminals):
        builder = MDPDialogBuilder(
            config_sample=self.config_sample,
            mdp_algorithm=algorithm,
            mdp_discard_states=discard_states,
            mdp_partial_assoc_rules=partial_assoc_rules,
            mdp_collapse_terminals=collapse_terminals)
        dialog = builder.build_dialog()

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

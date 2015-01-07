import os
import logging

import numpy as np

from ..policy import MDPDialogBuilder


logging.basicConfig(format="%(asctime)s:%(name)s:%(funcName)s:%(message)s",
                    level=logging.DEBUG)


class TestMDPDialogBuilder(object):

    def setup(self):
        tests_dir = os.path.abspath(os.path.dirname(__file__))
        csv_file = os.path.join(tests_dir, "titanic.csv")
        self.config_sample = np.genfromtxt(csv_file, skip_header=1,
                                           dtype=np.dtype(str),
                                           delimiter=",")
        print("")  # put a newline before the logging output

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

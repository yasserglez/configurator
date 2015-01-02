import os
import logging

import numpy as np

from ..policy import MDPDialogBuilder


logging.basicConfig(format="%(asctime)s:%(name)s:%(funcName)s:%(message)s",
                    level=logging.DEBUG)


class TestMDPDialogBuilder(object):

    def setup(self):
        tests_dir = os.path.abspath(os.path.dirname(__file__))
        csv_file = os.path.join(tests_dir, "Titanic.csv")
        self.config_sample = np.genfromtxt(csv_file, skip_header=1,
                                           dtype=np.dtype(str),
                                           delimiter=",")
        print("")  # put a newline before the logging output

    def _test_builder(self, mdp_algorithm):
        builder = MDPDialogBuilder(config_sample=self.config_sample,
                                   mdp_algorithm=mdp_algorithm)
        dialog = builder.build_dialog()

    def test_value_iteration_builder(self):
        self._test_builder("value-iteration")

    def test_policy_iteration_builder(self):
        self._test_builder("policy-iteration")

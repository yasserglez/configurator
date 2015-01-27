import sys
import random
import logging

import numpy as np

from .examples import load_email_client
from ..policy import DPDialogBuilder, RLDialogBuilder


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.handlers[0].setFormatter(logging.Formatter("%(asctime)s:%(message)s"))


class TestDPDialogBuilder(object):

    def setup(self):
        print("", file=sys.stderr)  # newline before the logging output
        self._email_client = load_email_client()

    def _test_builder(self, algorithm, discard_states,
                      partial_assoc_rules, collapse_terminals):
        builder = DPDialogBuilder(
            config_sample=self._email_client.config_sample,
            validate=True,
            assoc_rule_algorithm="apriori",
            assoc_rule_min_support=self._email_client.min_support,
            assoc_rule_min_confidence=self._email_client.min_confidence,
            dp_algorithm=algorithm,
            dp_discard_states=discard_states,
            dp_partial_assoc_rules=partial_assoc_rules,
            dp_collapse_terminals=collapse_terminals)
        dialog = builder.build_dialog()
        for var_index in self._email_client.questions:
            assert dialog.get_next_question() == var_index
            dialog.set_answer(var_index, self._email_client.config[var_index])
        assert dialog.is_complete()
        assert dialog.config == self._email_client.config

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


class TestRLDialogBuilder(object):

    def setup(self):
        print("", file=sys.stderr)  # newline before the logging output
        self._email_client = load_email_client(as_integers=True)
        seed = 42; random.seed(seed); np.random.seed(seed)

    def _test_builder(self, algorithm):
        builder = RLDialogBuilder(
            config_sample=self._email_client.config_sample,
            validate=True,
            assoc_rule_algorithm="apriori",
            assoc_rule_min_support=self._email_client.min_support,
            assoc_rule_min_confidence=self._email_client.min_confidence,
            rl_algorithm=algorithm)
        dialog = builder.build_dialog()
        for var_index in self._email_client.questions:
            assert dialog.get_next_question() == var_index
            dialog.set_answer(var_index, self._email_client.config[var_index])
        assert dialog.is_complete()
        assert dialog.config == self._email_client.config

    def test_qlearning(self):
        self._test_builder("q-learning")

    def test_sarsa(self):
        self._test_builder("sarsa")

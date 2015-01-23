import sys
import logging

from .examples import load_email_client
from ..policy import DPConfigDialogBuilder, RLConfigDialogBuilder


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.handlers[0].setFormatter(logging.Formatter("%(asctime)s:%(message)s"))


EMAIL_CLIENT = load_email_client()


class TestDPConfigDialogBuilder(object):

    def setup(self):
        print("", file=sys.stderr)  # newline before the logging output

    def _test_builder(self, algorithm, discard_states,
                      partial_assoc_rules, collapse_terminals):
        builder = DPConfigDialogBuilder(
            config_sample=EMAIL_CLIENT.config_sample,
            assoc_rule_algorithm="apriori",
            assoc_rule_min_support=EMAIL_CLIENT.min_support,
            assoc_rule_min_confidence=EMAIL_CLIENT.min_confidence,
            dp_algorithm=algorithm,
            dp_discard_states=discard_states,
            dp_partial_assoc_rules=partial_assoc_rules,
            dp_collapse_terminals=collapse_terminals,
            dp_validate=True)
        dialog = builder.build_dialog()
        for var_index in EMAIL_CLIENT.questions:
            assert dialog.get_next_question() == var_index
            dialog.set_answer(var_index, EMAIL_CLIENT.config[var_index])
        assert dialog.is_complete()
        assert dialog.config == EMAIL_CLIENT.config

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


class TestRLConfigDialogBuilder(object):

    def setup(self):
        print("", file=sys.stderr)  # newline before the logging output

    def _test_builder(self, algorithm):
        builder = RLConfigDialogBuilder(
            config_sample=EMAIL_CLIENT.config_sample,
            assoc_rule_algorithm="apriori",
            assoc_rule_min_support=EMAIL_CLIENT.min_support,
            assoc_rule_min_confidence=EMAIL_CLIENT.min_confidence,
            rl_algorithm=algorithm)
        dialog = builder.build_dialog()
        for var_index in EMAIL_CLIENT.questions:
            assert dialog.get_next_question() == var_index
            dialog.set_answer(var_index, EMAIL_CLIENT.config[var_index])
        assert dialog.is_complete()
        assert dialog.config == EMAIL_CLIENT.config

    def test_qlearning(self):
        self._test_builder("q-learning")

    def test_sarsa(self):
        self._test_builder("sarsa")

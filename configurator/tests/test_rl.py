from ..rl import RLDialogBuilder

from .common import BaseTest, load_email_client


class TestRLDialogBuilder(BaseTest):

    def setup(self):
        super().setup()
        self._email_client = load_email_client(as_integers=True)

    def _test_builder(self, algorithm, table):
        builder = RLDialogBuilder(
            config_sample=self._email_client.config_sample,
            validate=True,
            assoc_rule_algorithm="apriori",
            assoc_rule_min_support=self._email_client.min_support,
            assoc_rule_min_confidence=self._email_client.min_confidence,
            rl_algorithm=algorithm,
            rl_table=table)
        dialog = builder.build_dialog()
        for var_index in self._email_client.questions:
            assert dialog.get_next_question() == var_index
            dialog.set_answer(var_index, self._email_client.config[var_index])
        assert dialog.is_complete()
        assert dialog.config == self._email_client.config

    def test_qlearning_exact(self):
        self._test_builder("q-learning", "exact")

    def test_qlearning_approx(self):
        self._test_builder("q-learning", "approx")

    def test_sarsa_exact(self):
        self._test_builder("sarsa", "exact")

    def test_sarsa_approx(self):
        self._test_builder("sarsa", "approx")

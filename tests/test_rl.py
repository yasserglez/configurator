from configurator.rl import RLDialogBuilder


class TestRLDialogBuilder(object):

    def _test_builder(self, algorithm, table, email_client):
        builder = RLDialogBuilder(
            config_sample=email_client.config_sample,
            validate=True,
            assoc_rule_algorithm="apriori",
            assoc_rule_min_support=email_client.min_support,
            assoc_rule_min_confidence=email_client.min_confidence,
            rl_algorithm=algorithm,
            rl_table=table)
        dialog = builder.build_dialog()
        for var_index in email_client.questions:
            assert dialog.get_next_question() == var_index
            dialog.set_answer(var_index, email_client.config[var_index])
        assert dialog.is_complete()
        assert dialog.config == email_client.config

    def test_qlearning_exact(self, email_client):
        self._test_builder("q-learning", "exact", email_client)

    def test_qlearning_approx(self, email_client):
        self._test_builder("q-learning", "approx", email_client)

    def test_sarsa_exact(self, email_client):
        self._test_builder("sarsa", "exact", email_client)

    def test_sarsa_approx(self, email_client):
        self._test_builder("sarsa", "approx", email_client)

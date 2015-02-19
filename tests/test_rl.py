from configurator.rl import RLDialogBuilder


class TestRLDialogBuilder(object):

    def _test_builder(self, algorithm, table, table_features, email_client):
        builder = RLDialogBuilder(
            config_sample=email_client.config_sample,
            validate=True,
            rule_min_support=email_client.min_support,
            rule_min_confidence=email_client.min_confidence,
            rl_algorithm=algorithm,
            rl_table=table,
            rl_table_features=table_features)
        dialog = builder.build_dialog()
        for var_index in email_client.questions:
            assert dialog.get_next_question() == var_index
            dialog.set_answer(var_index, email_client.config[var_index])
        assert dialog.is_complete()
        assert dialog.config == email_client.config

    def _test_approx_builder(self, algorithm, table_features, email_client):
        self._test_builder(algorithm, "approx", table_features, email_client)

    def test_qlearning_exact(self, email_client):
        self._test_builder("q-learning", "exact", None, email_client)

    def test_sarsa_exact(self, email_client):
        self._test_builder("sarsa", "exact", None, email_client)

    def test_qlearning_approx_known_vars(self, email_client):
        self._test_approx_builder("q-learning", ["known-vars"], email_client)

    def test_qlearning_approx_last_answer(self, email_client):
        self._test_approx_builder("q-learning", ["last-answer"], email_client)

    def test_qlearning_approx_both(self, email_client):
        self._test_approx_builder("q-learning",
                                  ["known-vars", "last-answer"],
                                  email_client)

    def test_sarsa_approx_known_vars(self, email_client):
        self._test_approx_builder("sarsa", ["known-vars"], email_client)

    def test_sarsa_approx_last_answer(self, email_client):
        self._test_approx_builder("sarsa", ["last-answer"], email_client)

    def test_sarsa_approx_both(self, email_client):
        self._test_approx_builder("sarsa",
                                  ["known-vars", "last-answer"],
                                  email_client)

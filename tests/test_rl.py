from configurator.rl import RLDialogBuilder


class TestRLDialogBuilder(object):

    def _test_builder(self, algorithm, table, table_features, email_client):
        builder = RLDialogBuilder(domain=email_client.domain,
                                  rules=email_client.rules,
                                  sample=email_client.sample,
                                  rl_algorithm=algorithm,
                                  rl_table=table,
                                  rl_table_features=table_features,
                                  validate=True)
        dialog = builder.build_dialog()
        for var_index in email_client.questions:
            assert dialog.get_next_question() == var_index
            dialog.set_answer(var_index, email_client.config[var_index])
        assert dialog.is_complete()
        assert dialog.config == email_client.config

    def test_qlearning_exact(self, email_client):
        self._test_builder("q-learning", "exact", None, email_client)

    def test_sarsa_exact(self, email_client):
        self._test_builder("sarsa", "exact", None, email_client)

    def _test_approx_builder(self, algorithm, table_features, email_client):
        self._test_builder(algorithm, "approximate", table_features,
                           email_client)

    def test_qlearning_approx_known_variables(self, email_client):
        self._test_approx_builder("q-learning", ["known-variables"],
                                  email_client)

    def test_qlearning_approx_last_question(self, email_client):
        self._test_approx_builder("q-learning", ["last-question"],
                                  email_client)

    def test_qlearning_approx_all(self, email_client):
        self._test_approx_builder("q-learning",
                                  ["known-variables", "last-question"],
                                  email_client)

    def test_sarsa_approx_known_variables(self, email_client):
        self._test_approx_builder("sarsa", ["known-variables"], email_client)

    def test_sarsa_approx_last_question(self, email_client):
        self._test_approx_builder("sarsa", ["last-question"], email_client)

    def test_sarsa_approx_all(self, email_client):
        self._test_approx_builder("sarsa",
                                  ["known-variables", "last-question"],
                                  email_client)

from configurator.rl import RLDialogBuilder


class BaseTestRLDialogBuilder(object):

    def _create_builder(self, algorithm, table, table_features, email_client):
        raise NotImplementedError

    def _test_builder(self, algorithm, table, table_features, email_client):
        builder = self._create_builder(algorithm,
                                       table, table_features,
                                       email_client)
        dialog = builder.build_dialog()
        for config, num_questions in email_client.scenarios:
            dialog.reset()
            for i in range(num_questions):
                var_index = dialog.get_next_question()
                dialog.set_answer(var_index, config[var_index])
            assert dialog.is_complete()
            assert dialog.config == config

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


class TestRuleRLDialogBuilder(BaseTestRLDialogBuilder):

    def _create_builder(self, algorithm, table, table_features, email_client):
        return RLDialogBuilder(domain=email_client.domain,
                               rules=email_client.rules,
                               sample=email_client.sample,
                               rl_algorithm=algorithm,
                               rl_table=table,
                               rl_table_features=table_features,
                               validate=True)


class TestCSPRLDialogBuilder(BaseTestRLDialogBuilder):

    def _create_builder(self, algorithm, table, table_features, email_client):
        return RLDialogBuilder(domain=email_client.domain,
                               constraints=email_client.constraints,
                               sample=email_client.sample,
                               rl_algorithm=algorithm,
                               rl_table=table,
                               rl_table_features=table_features,
                               validate=True)

from configurator.rl import RLDialogBuilder


class BaseTestRLDialogBuilder(object):

    def _create_builder(self, algorithm, table, email_client):
        raise NotImplementedError

    def _test_builder(self, algorithm, table, email_client):
        builder = self._create_builder(algorithm, table, email_client)
        dialog = builder.build_dialog()
        for config, num_questions in email_client.scenarios:
            dialog.reset()
            for i in range(num_questions):
                var_index = dialog.get_next_question()
                dialog.set_answer(var_index, config[var_index])
            assert dialog.is_complete()
            assert dialog.config == config

    def test_qlearning_exact(self, email_client):
        self._test_builder("q-learning", "exact", email_client)

    def test_sarsa_exact(self, email_client):
        self._test_builder("sarsa", "exact", email_client)

    def test_qlearning_approximate(self, email_client):
        self._test_builder("q-learning", "approximate", email_client)

    def test_sarsa_approximate(self, email_client):
        self._test_builder("sarsa", "approximate", email_client)


class TestRuleRLDialogBuilder(BaseTestRLDialogBuilder):

    def _create_builder(self, algorithm, table, email_client):
        return RLDialogBuilder(var_domains=email_client.var_domains,
                               rules=email_client.rules,
                               sample=email_client.sample,
                               rl_algorithm=algorithm,
                               rl_table=table,
                               validate=True)


class TestCSPRLDialogBuilder(BaseTestRLDialogBuilder):

    def _create_builder(self, algorithm, table, email_client):
        return RLDialogBuilder(var_domains=email_client.var_domains,
                               constraints=email_client.constraints,
                               sample=email_client.sample,
                               rl_algorithm=algorithm,
                               rl_table=table,
                               validate=True)

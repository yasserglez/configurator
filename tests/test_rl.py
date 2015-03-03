from configurator.rl import RLDialogBuilder


class BaseTestRLDialogBuilder(object):

    def _test_builder(self, builder, email_client):
        dialog = builder.build_dialog()
        for config, num_questions in email_client.scenarios:
            dialog.reset()
            for i in range(num_questions):
                var_index = dialog.get_next_question()
                dialog.set_answer(var_index, config[var_index])
            assert dialog.is_complete()
            assert dialog.config == config


class TestRuleRLDialogBuilder(BaseTestRLDialogBuilder):

    def _create_builder(self, algorithm, table, email_client):
        return RLDialogBuilder(var_domains=email_client.var_domains,
                               rules=email_client.rules,
                               sample=email_client.sample,
                               rl_algorithm=algorithm,
                               rl_table=table,
                               validate=True)

    def test_qlearning_exact(self, email_client):
        builder = self._create_builder("q-learning", "exact", email_client)
        self._test_builder(builder, email_client)

    def test_sarsa_exact(self, email_client):
        builder = self._create_builder("sarsa", "exact", email_client)
        self._test_builder(builder, email_client)

    def test_qlearning_approximate(self, email_client):
        builder = self._create_builder("q-learning", "approximate",
                                       email_client)
        self._test_builder(builder, email_client)

    def test_sarsa_approximate(self, email_client):
        builder = self._create_builder("sarsa", "approximate", email_client)
        self._test_builder(builder, email_client)


class TestCSPRLDialogBuilder(BaseTestRLDialogBuilder):

    # TODO: Use a different example where enforcing local and global
    # consistency give different results.

    def _create_builder(self, algorithm, table, consistency, email_client):
        return RLDialogBuilder(var_domains=email_client.var_domains,
                               constraints=email_client.constraints,
                               sample=email_client.sample,
                               rl_algorithm=algorithm,
                               rl_table=table,
                               rl_consistency=consistency,
                               validate=True)

    def test_qlearning_exact_global(self, email_client):
        builder = self._create_builder("q-learning", "exact",
                                       "global", email_client)
        self._test_builder(builder, email_client)

    def test_sarsa_exact_global(self, email_client):
        builder = self._create_builder("sarsa", "exact", "global",
                                       email_client)
        self._test_builder(builder, email_client)

    def test_qlearning_approximate_global(self, email_client):
        builder = self._create_builder("q-learning", "approximate",
                                       "global", email_client)
        self._test_builder(builder, email_client)

    def test_sarsa_approximate_global(self, email_client):
        builder = self._create_builder("sarsa", "approximate",
                                       "global", email_client)
        self._test_builder(builder, email_client)

    def test_qlearning_exact_local(self, email_client):
        builder = self._create_builder("q-learning", "exact", "local",
                                       email_client)
        self._test_builder(builder, email_client)

    def test_sarsa_exact_local(self, email_client):
        builder = self._create_builder("sarsa", "exact", "local",
                                       email_client)
        self._test_builder(builder, email_client)

    def test_qlearning_approximate_local(self, email_client):
        builder = self._create_builder("q-learning", "approximate",
                                       "local", email_client)
        self._test_builder(builder, email_client)

    def test_sarsa_approximate_local(self, email_client):
        builder = self._create_builder("sarsa", "approximate",
                                       "local", email_client)
        self._test_builder(builder, email_client)

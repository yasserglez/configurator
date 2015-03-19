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

    def _create_builder(self, table, email_client):
        return RLDialogBuilder(email_client.var_domains,
                               email_client.sample,
                               rules=email_client.rules,
                               rl_table=table,
                               rl_num_episodes=50,
                               rl_rprop_epochs=100,
                               validate=True)

    def test_exact(self, email_client):
        builder = self._create_builder("exact", email_client)
        self._test_builder(builder, email_client)

    def test_approx(self, email_client):
        builder = self._create_builder("approx", email_client)
        self._test_builder(builder, email_client)


class TestCSPRLDialogBuilder(BaseTestRLDialogBuilder):

    def _create_builder(self, table, consistency, email_client):
        return RLDialogBuilder(email_client.var_domains,
                               email_client.sample,
                               constraints=email_client.constraints,
                               consistency=consistency,
                               rl_table=table,
                               rl_num_episodes=50,
                               rl_rprop_epochs=100,
                               validate=True)

    def test_exact_global(self, email_client):
        builder = self._create_builder("exact", "global", email_client)
        self._test_builder(builder, email_client)

    def test_approx_global(self, email_client):
        builder = self._create_builder("approx", "global", email_client)
        self._test_builder(builder, email_client)

    def test_exact_local(self, email_client):
        builder = self._create_builder("exact", "local", email_client)
        self._test_builder(builder, email_client)

    def test_approx_local(self, email_client):
        builder = self._create_builder("approx", "local", email_client)
        self._test_builder(builder, email_client)

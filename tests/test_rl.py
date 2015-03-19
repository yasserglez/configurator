import pytest

from configurator.rl import RLDialogBuilder


class _TestRLDialogBuilder(object):

    def _test_builder(self, builder, email_client):
        dialog = builder.build_dialog()
        for config, num_questions in email_client.scenarios:
            dialog.reset()
            for i in range(num_questions):
                var_index = dialog.get_next_question()
                dialog.set_answer(var_index, config[var_index])
            assert dialog.is_complete()
            assert dialog.config == config


class TestRuleRLDialogBuilder(_TestRLDialogBuilder):

    @pytest.mark.parametrize("rl_table", ("exact", "approx"))
    def test_build_dialog(self, rl_table, email_client):
        builder = RLDialogBuilder(email_client.var_domains,
                                  email_client.sample,
                                  rules=email_client.rules,
                                  rl_num_episodes=50,
                                  rl_table=rl_table,
                                  rl_rprop_epochs=100,
                                  validate=True)
        self._test_builder(builder, email_client)


class TestCSPRLDialogBuilder(_TestRLDialogBuilder):

    @pytest.mark.parametrize(("consistency", "rl_table"),
                             (("global", "exact"),
                              ("global", "approx"),
                              ("local", "exact"),
                              ("local", "approx")))
    def test_build_dialog(self, consistency, rl_table, email_client):
        builder = RLDialogBuilder(email_client.var_domains,
                                  email_client.sample,
                                  constraints=email_client.constraints,
                                  consistency=consistency,
                                  rl_num_episodes=50,
                                  rl_table=rl_table,
                                  rl_rprop_epochs=100,
                                  validate=True)
        self._test_builder(builder, email_client)

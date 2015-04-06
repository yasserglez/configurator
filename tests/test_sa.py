import pytest

from configurator.sequence.sa import SADialogBuilder


class _TestSADialogBuilder(object):

    def _test_builder(self, builder, email_client):
        dialog = builder.build_dialog()
        for config, num_questions in email_client.scenarios:
            dialog.reset()
            for i in range(num_questions):
                var_index = dialog.get_next_question()
                dialog.set_answer(var_index, config[var_index])
            assert dialog.is_complete()
            assert dialog.config == config


class TestRuleSADialogBuilder(_TestSADialogBuilder):

    @pytest.mark.parametrize("initial_solution", ("random", "degree"))
    def test_build_dialog(self, initial_solution, email_client):
        builder = SADialogBuilder(email_client.var_domains,
                                  email_client.sample,
                                  rules=email_client.rules,
                                  total_episodes=100,
                                  eval_episodes=10,
                                  initial_solution=initial_solution,
                                  validate=True)
        self._test_builder(builder, email_client)


class TestCSPSADialogBuilder(_TestSADialogBuilder):

    @pytest.mark.parametrize(("consistency", "initial_solution"),
                             (("global", "random"),
                              ("global", "degree"),
                              ("local", "random"),
                              ("local", "degree")))
    def test_build_dialog(self, consistency, initial_solution, email_client):
        builder = SADialogBuilder(email_client.var_domains,
                                  email_client.sample,
                                  constraints=email_client.constraints,
                                  consistency=consistency,
                                  total_episodes=100,
                                  eval_episodes=10,
                                  initial_solution=initial_solution,
                                  validate=True)
        self._test_builder(builder, email_client)

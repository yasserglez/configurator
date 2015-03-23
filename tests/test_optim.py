import pytest

from configurator.optim import OptimDialogBuilder


class _TestOptimDialogBuilder(object):

    def _test_builder(self, builder, email_client):
        dialog = builder.build_dialog()
        for config, num_questions in email_client.scenarios:
            dialog.reset()
            for i in range(num_questions):
                var_index = dialog.get_next_question()
                dialog.set_answer(var_index, config[var_index])
            assert dialog.is_complete()
            assert dialog.config == config


class TestRuleOptimDialogBuilder(_TestOptimDialogBuilder):

    def test_build_dialog(self, email_client):
        builder = OptimDialogBuilder(email_client.var_domains,
                                     email_client.sample,
                                     rules=email_client.rules,
                                     num_episodes=50,
                                     eval_batch=5,
                                     validate=True)
        self._test_builder(builder, email_client)


class TestCSPOptimDialogBuilder(_TestOptimDialogBuilder):

    @pytest.mark.parametrize("consistency", ("global", "local"))
    def test_build_dialog(self, consistency, email_client):
        builder = OptimDialogBuilder(email_client.var_domains,
                                     email_client.sample,
                                     constraints=email_client.constraints,
                                     consistency=consistency,
                                     num_episodes=50,
                                     eval_batch=5,
                                     validate=True)
        self._test_builder(builder, email_client)

import pytest

from configurator.dialogs import Dialog
from configurator.dialogs.rl import RLDialogBuilder


class TestRLDialog(object):

    @pytest.mark.parametrize(("use_rules", "table"),
                             ((True, "exact"),
                              (True, "approx"),
                              (False, "exact"),
                              (False, "approx")),
                             ids=("rules-exact",
                                  "rules-approx",
                                  "constraints-exact",
                                  "constraints-approx"))
    def test_save(self, tmpdir, use_rules, table, email_client):
        rules = email_client.rules if use_rules else None
        constraints = None if use_rules else email_client.constraints
        builder = RLDialogBuilder(email_client.var_domains,
                                  email_client.sample,
                                  rules=rules,
                                  constraints=constraints,
                                  total_episodes=100,
                                  table=table)
        saved_dialog = builder.build_dialog()
        file_path = str(tmpdir.join("dialog.zip"))
        saved_dialog.save(file_path)
        new_dialog = Dialog.load(file_path)
        assert new_dialog.var_domains == saved_dialog.var_domains
        new_dialog.reset()
        assert new_dialog.get_next_question() == 1
        new_dialog.set_answer(1, "lgi")
        assert new_dialog.config == {0: "no", 1: "lgi"}


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

    @pytest.mark.parametrize("table", ("exact", "approx"))
    def test_build_dialog(self, table, email_client):
        builder = RLDialogBuilder(email_client.var_domains,
                                  email_client.sample,
                                  rules=email_client.rules,
                                  total_episodes=100,
                                  table=table,
                                  validate=True)
        self._test_builder(builder, email_client)


class TestCSPRLDialogBuilder(_TestRLDialogBuilder):

    @pytest.mark.parametrize(("consistency", "table"),
                             (("global", "exact"),
                              ("global", "approx"),
                              ("local", "exact"),
                              ("local", "approx")))
    def test_build_dialog(self, consistency, table, email_client):
        builder = RLDialogBuilder(email_client.var_domains,
                                  email_client.sample,
                                  constraints=email_client.constraints,
                                  consistency=consistency,
                                  total_episodes=100,
                                  table=table,
                                  validate=True)
        self._test_builder(builder, email_client)

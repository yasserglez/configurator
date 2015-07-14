import pytest

from configurator.dialogs import Dialog
from configurator.dialogs.ga import GADialogBuilder, GADialog


class TestGADialog(object):

    @pytest.mark.parametrize("use_rules", (True, False),
                             ids=("rules", "constraints"))
    def test_get_next_question(self, use_rules, email_client):
        rules = email_client.rules if use_rules else None
        constraints = None if use_rules else email_client.constraints
        dialog = GADialog(email_client.var_domains, [1, 0],
                          rules=rules,
                          constraints=constraints,
                          validate=True)
        dialog.reset()
        assert dialog.get_next_question() == 1
        dialog.set_answer(1, "lgi")
        assert dialog.is_complete()

    @pytest.mark.parametrize("use_rules", (True, False),
                             ids=("rules", "constraints"))
    def test_save(self, tmpdir, use_rules, email_client):
        rules = email_client.rules if use_rules else None
        constraints = None if use_rules else email_client.constraints
        saved_dialog = GADialog(email_client.var_domains, [1, 0],
                                rules=rules,
                                constraints=constraints)
        file_path = str(tmpdir.join("dialog.zip"))
        saved_dialog.save(file_path)
        new_dialog = Dialog.load(file_path)
        assert new_dialog.var_domains == saved_dialog.var_domains
        assert new_dialog.var_perm == saved_dialog.var_perm
        new_dialog.reset()
        assert new_dialog.get_next_question() == 1
        new_dialog.set_answer(1, "lgi")
        assert new_dialog.config == {0: "no", 1: "lgi"}


class _TestGADialogBuilder(object):

    def _test_builder(self, builder, email_client):
        dialog = builder.build_dialog()
        for config, num_questions in email_client.scenarios:
            dialog.reset()
            for i in range(num_questions):
                var_index = dialog.get_next_question()
                dialog.set_answer(var_index, config[var_index])
            assert dialog.is_complete()
            assert dialog.config == config


class TestRuleGADialogBuilder(_TestGADialogBuilder):

    @pytest.mark.parametrize("initial_solution", ("random", "degree"))
    def test_build_dialog(self, initial_solution, email_client):
        builder = GADialogBuilder(email_client.var_domains,
                                  email_client.sample,
                                  rules=email_client.rules,
                                  total_episodes=1000,
                                  eval_episodes=10,
                                  population_size=5,
                                  initial_solution=initial_solution,
                                  validate=True)
        self._test_builder(builder, email_client)


class TestCSPGADialogBuilder(_TestGADialogBuilder):

    @pytest.mark.parametrize(("consistency", "initial_solution"),
                             (("global", "random"),
                              ("global", "degree"),
                              ("local", "random"),
                              ("local", "degree")))
    def test_build_dialog(self, consistency, initial_solution, email_client):
        builder = GADialogBuilder(email_client.var_domains,
                                  email_client.sample,
                                  constraints=email_client.constraints,
                                  consistency=consistency,
                                  total_episodes=1000,
                                  eval_episodes=10,
                                  population_size=5,
                                  initial_solution=initial_solution,
                                  validate=True)
        self._test_builder(builder, email_client)

import pytest

from configurator.dialogs import Dialog
from configurator.sequence import PermutationDialog


class TestPermutationDialog(object):

    @pytest.mark.parametrize("use_rules", (True, False),
                             ids=("rules", "constraints"))
    def test_get_next_question(self, use_rules, email_client):
        rules = email_client.rules if use_rules else None
        constraints = None if use_rules else email_client.constraints
        dialog = PermutationDialog(email_client.var_domains, [1, 0],
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
        saved_dialog = PermutationDialog(email_client.var_domains, [1, 0],
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
